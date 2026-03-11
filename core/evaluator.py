import spacy
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from core.metrics import flag_gibberish, count_syllables, calculate_overlap_metrics
from lexical_diversity import lex_div as ld

class ConsistencyValidator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nlp = spacy.load("en_core_web_trf")
        
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        
        self.nli_classifier = CrossEncoder("cross-encoder/nli-deberta-v3-small", device=self.device)

    def evaluate_coherence(self, content: str, window=2) -> float:
        segments = content.split(". ")
        if len(segments) < 2:
            return 1.0

        vecs = self.embedder.encode(segments, convert_to_numpy=True)
        sim_scores = []
        for i in range(len(segments) - window + 1):
            block = vecs[i : i + window]
            for j in range(len(block) - 1):
                from scipy.spatial.distance import cosine
                sim_scores.append(1 - cosine(block[j], block[j + 1]))

        return float(np.mean(sim_scores)) if sim_scores else 0.0

    def check_nli_entailment(self, premise: str, hypothesis: str) -> dict:
        scores = self.nli_classifier.predict([(premise, hypothesis)])[0]
        
        labels = ["Contradiction", "Entailment", "Neutral"]
        probabilities = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()
        
        return {
            "dominant_label": labels[np.argmax(probabilities)],
            "contradiction_prob": float(probabilities[0]),
            "entailment_prob": float(probabilities[1]),
            "neutral_prob": float(probabilities[2])
        }

    def analyze(self, input_context: str, generated_output: str) -> dict:
        doc = self.nlp(generated_output)
        raw_tokens = [t.text for t in doc if t.is_alpha]
        
        gibberish_arr = [1 if flag_gibberish(t) else 0 for t in raw_tokens]
        gibb_score = np.mean(gibberish_arr) if gibberish_arr else 0.0
        
        overlap_stats = calculate_overlap_metrics(input_context, generated_output, self.nlp)
        nli_stats = self.check_nli_entailment(input_context, generated_output)
        coherence_val = self.evaluate_coherence(generated_output)

        return {
            "structural_metrics": {
                "gibberish_ratio": round(gibb_score, 4),
                "lexical_diversity_ttr": round(ld.ttr(raw_tokens), 4) if raw_tokens else 0.0,
                "coherence_score": round(coherence_val, 4)
            },
            "factual_alignment": {
                "token_overlap": overlap_stats["token_overlap_ratio"],
                "entity_overlap": overlap_stats["entity_overlap_ratio"]
            },
            "nli_inference": nli_stats
        }

def synthesize_diagnostic_report(metrics_payload: dict) -> str:
    summary_statements = []
    
    nli_data = metrics_payload.get("nli_inference", {})
    primary_verdict = nli_data.get("dominant_label", "Neutral")
    contradict_risk = nli_data.get("contradiction_prob", 0.0)
    
    structural_data = metrics_payload.get("structural_metrics", {})
    gibberish_index = structural_data.get("gibberish_ratio", 0.0)
    coherence_index = structural_data.get("coherence_score", 0.0)
    
    alignment_data = metrics_payload.get("factual_alignment", {})
    entity_retention = alignment_data.get("entity_overlap", 0.0)

    if primary_verdict == "Entailment":
        summary_statements.append("The generated response is factually grounded and directly supported by your source context.")
    elif primary_verdict == "Contradiction" or contradict_risk > 0.4:
        summary_statements.append("CRITICAL WARNING: The model has generated claims that actively contradict the provided source material.")
    else:
        summary_statements.append("The output is neutral; it contains claims that are neither explicitly confirmed nor denied by the source text, which may indicate a mild hallucination.")

    if entity_retention < 0.3 and primary_verdict != "Contradiction":
        summary_statements.append("It drops a significant amount of key terminology from the source, suggesting it might be losing focus or over-summarizing.")

    if gibberish_index > 0.15:
        summary_statements.append("Additionally, the text exhibits repetitive character patterns or unnatural token sequences typical of AI degradation.")
    elif coherence_index < 0.6:
        summary_statements.append("The logical flow between sentences is highly disjointed, making it difficult to read.")
    else:
        summary_statements.append("Structurally, the text is coherent and free of obvious synthetic artifacts.")

    return " ".join(summary_statements)
