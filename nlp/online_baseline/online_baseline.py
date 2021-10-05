from functools import lru_cache
import argparse
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from pymystem3 import Mystem
from sklearn.neighbors import KDTree



TOKENIZATION_TYPE='sentencepiece'

ALLOWED_ALPHABET=list(map(chr, range(ord('а'), ord('я') + 1)))
ALLOWED_ALPHABET.extend(map(chr, range(ord('a'), ord('z') + 1)))
ALLOWED_ALPHABET.extend(list(map(str.upper, ALLOWED_ALPHABET)))
ALLOWED_ALPHABET = set(ALLOWED_ALPHABET)


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


def calc_embs(words):
    words = ' '.join(map(normalize, words))
    inds = get_w2v_indicies(words)
    return [None if i is None else embs_vectors[i] for i in inds]


@lru_cache()
def find_closest_nontoxic(word, threshold=0.5, allow_self=False):
    if toxicity.get(word, 1.0) <= threshold:
        return word
    
    if word not in toxicity and word not in embs_voc:
        return None
    
    threshold = min(toxicity.get(word, threshold), threshold)
    word = normalize(word)
    word_emb = calc_embs([word])
    if word_emb is None or word_emb[0] is None:
        return None
    
    for i in embs_tree.query(word_emb)[1][0]:
        other_word = embs_voc_by_id[nontoxic_emb_inds[i]]
        if (other_word != word or allow_self) and toxicity.get(other_word, 1.0) <= threshold:
            return other_word
    return None


def load_embeddings(path):
    embs_file = np.load(path, allow_pickle=True)
    embs_vectors = embs_file['vectors']
    embs_voc = embs_file['voc'].item()
    embs_vectors_normed = embs_vectors / np.linalg.norm(embs_vectors, axis=1, keepdims=True)

    embs_voc_by_id = [None for i in range(len(embs_vectors))]
    for word, idx in embs_voc.items():
        if embs_voc_by_id[idx] is None:
            embs_voc_by_id[idx] = word
    
    nontoxic_emb_inds = [ind for word, ind in embs_voc.items() if toxicity.get(word, 1.0) <= 0.5]
    embs_vectors_normed_nontoxic = embs_vectors_normed[nontoxic_emb_inds]
    
    embs_tree = KDTree(embs_vectors_normed_nontoxic, leaf_size=20)
    
    return embs_vectors, embs_voc, embs_voc_by_id, nontoxic_emb_inds, embs_vectors_normed_nontoxic, embs_tree


def detox(line):
    words = normalize(line).split()
    fixed_words = [find_closest_nontoxic(word, allow_self=True) or '' for word in words]
    return ' '.join(fixed_words)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_texts', type=argparse.FileType('r'))
    parser.add_argument('fixed_texts', type=argparse.FileType('w'))
    parser.add_argument('--data', type=argparse.FileType('rb'), required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--tokenizer', required=True, type=str)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # load precomputed toxicities
    with args.data as f:
        toxicity = pkl.load(f)
        nontoxic_emb_inds = pkl.load(f)
        
    # load embeddings
    embs_vectors, embs_voc, embs_voc_by_id, nontoxic_emb_inds, embs_vectors_normed_nontoxic, embs_tree = load_embeddings(args.embeddings)
    
    # initialize stemer
    stemmer = Mystem()
    
    with args.original_texts, args.fixed_texts:
        for line in tqdm(args.original_texts):
            print(detox(line.strip()), file=args.fixed_texts)
