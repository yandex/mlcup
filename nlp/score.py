from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import softmax, sigmoid
from tqdm.auto import tqdm
import numpy as np
import argparse
from pymystem3 import Mystem


TOXIC_CLASS=-1
TOKENIZATION_TYPE='sentencepiece'


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
    sentence = ''.join(map(lambda c: c if c.isalpha() else ' ', sentence.lower()))
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
    tqdm_fn = tqdm if verbose else lambda x, total: x
    
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
            res.append(embs_voc[w])
        else:
            lemma = stemmer.lemmatize(w)[0]
            res.append(embs_voc.get(lemma, None))
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
    return [None if i is None else embs_vectors[i] for i in inds]


def count_none(array):
    res = 0
    for el in array:
        if el is None:
            res += 1
    return res


def greedy_match_embs(a, b, dots=None):
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len([x for x in a if x is not None])
    # compute dot-product on initial run
    if dots is None:
        a_none_count = count_none(a)
        b_none_count = count_none(b)
        
        if a_none_count + b_none_count > 0:
            # None values don't match anything except other None values
            return max(b_none_count - a_none_count, 0) + greedy_match_embs(
                [x for x in a if x is not None],
                [x for x in b if x is not None]
            )
        # scale embeddings so that their dot product turns into cosine
        a = np.array(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.array(b) / np.linalg.norm(b, axis=1, keepdims=True)
        dots = np.dot(a, b.T)
    # select the closest embeddings
    # note: assume None embeddings are filtered out at this point
    a_closest, b_closest = np.unravel_index(np.argmax(dots), dots.shape)
    min_dist = (1 - dots[a_closest, b_closest]) / 2
    
    # exclude the matched embeddings from the subsequent iterations
    remaining_a_inds = np.arange(len(a)) != a_closest
    remaining_b_inds = np.arange(len(b)) != b_closest
    
    return min_dist + greedy_match_embs(
        a[remaining_a_inds], 
        b[remaining_b_inds], 
        dots[remaining_a_inds][:, remaining_b_inds]
    )


def calc_semantic_distance(a, b):
    a_embs = calc_embs(a)
    b_embs = calc_embs(b)
    return np.maximum(greedy_match_embs(a_embs, b_embs), 0)


def distance_score(original, fixed):
    original = original.split()
    fixed = fixed.split()
    
    return calc_semantic_distance(original, fixed) / len(original)


def compute_score(original_sentences, fixed_sentences, threshold=0.5, batch_size=5):
    fixed_preds = predict_toxicity(fixed_sentences, threshold=threshold, batch_size=batch_size)
    
    scores = []
    for original_sentence, fixed_sentence, fixed_pred in tqdm(zip(
        original_sentences, fixed_sentences, fixed_preds
    )):
        original_sentence = normalize(original_sentence)
        fixed_sentence = normalize(fixed_sentence)
        if fixed_pred:
            score = 1
        else:
            score = distance_score(original_sentence, fixed_sentence)
        scores.append(score)
    return np.mean(scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_texts', type=argparse.FileType('r'))
    parser.add_argument('fixed_texts', type=argparse.FileType('r'))
    parser.add_argument('--score', type=argparse.FileType('w'))
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--embeddings', type=str, required=True)
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

    print("Loading embeddings", file=sys.stderr)
    embs_vectors, embs_voc, embs_voc_by_id = load_embeddings(args.embeddings)
    
    with torch.inference_mode(True):
        print("Scoring", file=sys.stderr)
        print(100 * (1 - compute_score(original_texts, fixed_texts)), file=args.score)


