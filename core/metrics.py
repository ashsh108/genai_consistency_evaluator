import re
import string
import numpy as np
import spacy
import nltk
from nltk.corpus import cmudict, words
from metaphone import doublemetaphone
from scipy.spatial.distance import cosine
from rapidfuzz import process, fuzz
from lexical_diversity import lex_div as ld

nltk.download('cmudict', quiet=True)
nltk.download('words', quiet=True)

try:
    cmu_lookup = cmudict.dict()
except LookupError:
    cmu_lookup = {}

english_vocab = set(words.words())

def count_syllables(term: str) -> int:
    term_lower = term.lower()
    if term_lower in cmu_lookup:
        return max([len([y for y in x if y[-1].isdigit()]) for x in cmu_lookup[term_lower]])
    return 1

def flag_gibberish(term: str) -> bool:
    if term.lower() in english_vocab:
        return False
        
    repetitive = bool(re.match(r"^(.)\1{3,}$", term))
    common_pairs = {"th", "he", "in", "er", "an", "re", "on", "at", "en", "nd"}
    pairs = [term[i:i+2] for i in range(len(term)-1)]
    uncommon_ratio = sum(1 for p in pairs if p not in common_pairs) / max(len(pairs), 1)
    
    no_vowels = not re.search(r"[aeiouy]", term, re.IGNORECASE)
    heavy_symbols = sum(1 for c in term if c in string.punctuation or c.isdigit()) > len(term) * 0.4
    mixed_caps = sum(1 for c in term if c.isupper()) not in {0, len(term)}
    
    return repetitive or (uncommon_ratio > 0.6) or no_vowels or heavy_symbols or mixed_caps

def calculate_overlap_metrics(source_text: str, target_text: str, nlp_engine) -> dict:
    def extract_components(text_block):
        processed = nlp_engine(text_block)
        tokens = {t.lemma_.lower() for t in processed if not t.is_stop}
        entities = {e.text.lower() for e in processed.ents}
        return tokens, entities

    src_tokens, src_ents = extract_components(source_text)
    tgt_tokens, tgt_ents = extract_components(target_text)

    def compute_fuzzy_intersect(set_a, set_b, threshold=85):
        matches = set()
        for item in set_a:
            match_res = process.extractOne(item, set_b, scorer=fuzz.ratio)
            if match_res and match_res[1] >= threshold:
                matches.add(match_res[0])
        return matches

    tok_overlap = len(compute_fuzzy_intersect(src_tokens, tgt_tokens)) / len(tgt_tokens) if tgt_tokens else 0.0
    ent_overlap = len(compute_fuzzy_intersect(src_ents, tgt_ents)) / len(tgt_ents) if tgt_ents else 0.0

    return {
        "token_overlap_ratio": round(tok_overlap, 4),
        "entity_overlap_ratio": round(ent_overlap, 4)
    }
