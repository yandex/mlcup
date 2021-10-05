# encoding=utf-8
import numpy as np

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import softmax, sigmoid
from tqdm.auto import tqdm
import argparse
from pymystem3 import Mystem
import sys
from collections import Counter
from functools import partial
import kenlm


TOXIC_CLASS=-1
TOKENIZATION_TYPE='sentencepiece'


ALLOWED_ALPHABET=list(map(chr, range(ord('а'), ord('я') + 1)))
ALLOWED_ALPHABET.extend(map(chr, range(ord('a'), ord('z') + 1)))
ALLOWED_ALPHABET.extend(list(map(str.upper, ALLOWED_ALPHABET)))
ALLOWED_ALPHABET = set(ALLOWED_ALPHABET)


def logits_to_toxic_probas(logits):
    if logits.shape[-1] > 1:
        activation = lambda x: softmax(x, -1)
    else:
        activation = sigmoid
    return activation(logits)[:, TOXIC_CLASS].cpu().detach().numpy()


def is_word_start(token):
    if TOKENIZATION_TYPE == 'sentencepiece':
        return token.startswith('▁')
    if TOKENIZATION_TYPE == 'bert':
        return not token.startswith('##')
    raise ValueError("Unknown tokenization type")


def normalize(sentence, max_tokens_per_word=20):
    def validate_char(c):
        return c in ALLOWED_ALPHABET
    
    sentence = ''.join(map(lambda c: c if validate_char(c) else ' ', sentence.lower()))
    ids = tokenizer(sentence)['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(ids)[1:-1]
    
    result = []
    num_continuation_tokens = 0
    for token in tokens:
        if not is_word_start(token):
            num_continuation_tokens += 1
            if num_continuation_tokens < max_tokens_per_word:
                result.append(token.lstrip('#▁'))
        else:
            num_continuation_tokens = 0
            result.extend([' ', token.lstrip('▁#')])
    
    return ''.join(result).strip()


def iterate_batches(data, batch_size=40):
    batch = []
    for x in data:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def predict_toxicity(sentences, batch_size=5, threshold=0.5, return_scores=False, verbose=True):
    results = []
    tqdm_fn = partial(tqdm, miniters=50) if verbose else lambda x, total: x
    
    for batch in tqdm_fn(iterate_batches(sentences, batch_size), total=np.ceil(len(sentences) / batch_size)):
        normlized = [normalize(sent, max_tokens_per_word=5) for sent in batch]
        tokenized = tokenizer(normlized, return_tensors='pt', padding=True, max_length=512, truncation=True)
        
        logits = model(**{key: val.to(model.device) for key, val in tokenized.items()}).logits
        preds = logits_to_toxic_probas(logits)
        if not return_scores:
            preds = preds >= threshold
        results.extend(preds)
    return results


def get_w2v_indicies(a):
    res = []
    if isinstance(a, str):
        a = a.split()
    for w in a:
        if w in embs_voc:
            res.append((w, embs_voc[w]))
        else:
            for lemma in stemmer.lemmatize(w):
                if lemma.isalpha():
                    res.append((w, embs_voc.get(lemma, None)))
    return res


def load_embeddings(path):
    embs_file = np.load(path, allow_pickle=True)
    embs_vectors = embs_file['vectors']
    embs_voc = embs_file['voc'].item()

    embs_voc_by_id = [None for i in range(len(embs_vectors))]
    for word, idx in embs_voc.items():
        if embs_voc_by_id[idx] is None:
            embs_voc_by_id[idx] = word
    return embs_vectors, embs_voc, embs_voc_by_id


def calc_embs(words):
    words = ' '.join(map(normalize, words))
    inds = get_w2v_indicies(words)
    return [(w, i if i is None else embs_vectors[i]) for w, i in inds]


def calc_single_embedding_dist(a, b):
    a_s, a_v = a
    b_s, b_v = b
    if a_s == b_s:
        return 0.0
    if a_v is None or b_v is None:
        return 1.0
    a = a_v
    b = b_v
    # inexact match is punished by 0.1
    return 0.1 + 0.9 * (1 - a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)) / 2


def greedy_match_embs(a, b, max_dist=99999, cache=None, a_ind=0, b_ind=0):
    a_len = len(a) - a_ind
    b_len = len(b) - b_ind
    minlen = min(a_len, b_len)
    maxlen = max(a_len, b_len)
    if minlen == 0:
        return np.minimum(maxlen, max_dist)
    if maxlen - minlen >= max_dist:
        return max_dist
    
    if cache is None:
        cache = {}
    
    cache_key = (a_len, b_len)
    if cache_key in cache:
        return cache[cache_key]
        
    min_dist = max_dist
    
    first_dist = calc_single_embedding_dist(a[a_ind], b[b_ind])
    if max_dist >= first_dist:
        min_dist = np.minimum(min_dist, first_dist + greedy_match_embs(
            a, b, max_dist, cache, a_ind + 1, b_ind + 1
        ))
    
    if first_dist > 0 and max_dist >= 1:
        min_dist = np.minimum(min_dist, 1 + greedy_match_embs(
            a, b, max_dist, cache, a_ind + 1, b_ind
        ))
        min_dist = np.minimum(min_dist, 1 + greedy_match_embs(
            a, b, max_dist, cache, a_ind, b_ind + 1
        ))
    
    cache[cache_key] = min_dist
    
    return min_dist



def calc_semantic_distance(a, b):
    a_embs = calc_embs(a)
    b_embs = calc_embs(b)
    
    clip_distance = 5  # this clips long computations
    return np.exp(-(greedy_match_embs(a_embs, b_embs, max_dist=clip_distance) / (0.6 * np.log(1 + len(a)))) ** 2)


def distance_score(original, fixed):
    original = original.split()
    fixed = fixed.split()
    
    return calc_semantic_distance(original, fixed)


def compute_lmdiff(original, fixed):
    original_lm_logproba = lm.score(original, bos=True, eos=True)
    fixed_lm_logproba = lm.score(fixed, bos=True, eos=True)
    
    probability_fraction = 10**((fixed_lm_logproba - original_lm_logproba) / 25)
    
    return np.clip(probability_fraction, 0.0, 1.0)


def compute_score(original_sentences, fixed_sentences, threshold=0.5, batch_size=5):
    fixed_toxicities = predict_toxicity(fixed_sentences, threshold=threshold, batch_size=batch_size, return_scores=True)
    scores = []
    lmdiffs = []
    emb_dists = []
    for original_sentence, fixed_sentence, fixed_toxicity in tqdm(zip(
        original_sentences, fixed_sentences, fixed_toxicities
    ), miniters=250):
        original_sentence = normalize(original_sentence)
        fixed_sentence = normalize(fixed_sentence)
        
        distance = distance_score(original_sentence, fixed_sentence)
        lmdiff = compute_lmdiff(original_sentence, fixed_sentence)
        
        score = (1 - fixed_toxicity) * distance * lmdiff
        
        lmdiffs.append(lmdiff)
        emb_dists.append(distance)
        scores.append(score)
    
    print('average toxicity:', np.mean(fixed_toxicities), file=sys.stderr)
    print('mean lmdiff:', np.mean(lmdiffs), file=sys.stderr)
    print('mean distance_score:', np.mean(emb_dists), file=sys.stderr)
    
    return np.mean(scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_texts', type=argparse.FileType('r'))
    parser.add_argument('fixed_texts', type=argparse.FileType('r'))
    parser.add_argument('--score', type=argparse.FileType('w'))
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--lm', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print("Loading tokenizer", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Loading model", file=sys.stderr)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device)
    
    print("Loading texts", file=sys.stderr)
    # load reference and submitted comments
    original_texts = list(map(str.strip, args.original_texts))
    fixed_texts = list(map(str.strip, args.fixed_texts))
    
    assert len(original_texts) == len(fixed_texts)

    stemmer = Mystem()
    
    print("Loading LM", file=sys.stderr)
    lm = kenlm.Model(args.lm)

    print("Loading embeddings", file=sys.stderr)
    embs_vectors, embs_voc, embs_voc_by_id = load_embeddings(args.embeddings)
    
    with torch.inference_mode(True):
        print("Scoring", file=sys.stderr)
        print("{:.2f}".format(100 * compute_score(original_texts, fixed_texts)), file=args.score)

