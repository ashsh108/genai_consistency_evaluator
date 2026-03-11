# genai_consistency_evaluator
This repository provides a deterministic, non-generative evaluation pipeline to assess the factual consistency, semantic overlap, and structural integrity of Large Language Model (LLM) outputs.

Rather than relying on another LLM to grade outputs (which introduces secondary hallucination risks), this system utilizes deterministic NLP metrics, Cross-Encoder Natural Language Inference (NLI), and lexical analysis to ground the generated text against a known source context.

## Key Features
Factual Alignment (NLI): Uses a DeBERTa-based Cross-Encoder to classify the relationship between the source context and the generated output as Entailment, Contradiction, or Neutral.

Semantic & Entity Overlap: Calculates fuzzy matching scores for critical tokens and Named Entities (NER) using spaCy and RapidFuzz, ensuring key facts are not dropped.

Structural Integrity Analysis: Detects AI-generated gibberish, measures lexical diversity (TTR), and computes sentence-to-sentence coherence using SentenceTransformers.

Decoupled Architecture: Features a standalone FastAPI backend for programmatic integration and a separate Streamlit frontend for visual interaction and testing.


## Code Base
requirements.txt
The Blueprint
This file lists every external Python library the project needs to run. By defining exact dependencies (like fastapi, spacy, and sentence-transformers), it guarantees that the code will execute exactly the same way on your machine as it does on a production server.

metrics.py
The Calculation Engine
This file houses all the raw, mathematical, and rule-based functions. It handles the low-level text parsing that does not require heavy neural networks, such as counting syllables, detecting gibberish via character repetition, and performing fuzzy string matching to calculate token overlap. Keeping these separate ensures the code remains fast and easy to test.

evaluator.py
The Brain
This is the heavy lifter. It initializes and stores the large machine learning models (the spaCy NLP pipeline, the SentenceTransformer for coherence, and the DeBERTa Cross-Encoder for Natural Language Inference) in memory. It orchestrates the entire evaluation process, taking the raw text, passing it through the models and the metrics.py functions, and compiling the final scores into a structured dictionary.

api.py
The Communication Layer
This script acts as the server. Using FastAPI, it wraps the logic from evaluator.py into a RESTful web endpoint. It waits for incoming HTTP requests containing the context and generated text, processes them using the evaluator, and returns the results as a clean JSON response. This allows any other application to use your evaluation logic programmatically.

app.py
The User Interface
This is the visual dashboard built with Streamlit. It acts as a client that talks to your API. It provides text boxes for human users to paste their source context and generated text, sends that data to the API running on port 8080, and then translates the raw JSON response into readable metrics, charts, and warnings on the screen.
