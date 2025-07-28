from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
import torch
from evaluate import load as load_metric
from typing import List
from model_utils import (
    generate_text,
    summarize_text,
    check_similarity,
    run_filters,
    compute_perplexity
)

app = FastAPI()

# === Modèles et pipelines ===
generation_tokenizer = AutoTokenizer.from_pretrained("gpt2")
generation_model = GPT2LMHeadModel.from_pretrained("gpt2")
summarizer = pipeline("summarization", model="facebook/bart-base")
classifier = pipeline("text-classification", model="unitary/toxic-bert")
rouge_metric = load_metric("rouge")


class PromptRequest(BaseModel):
    prompt: str

class SimilarityRequest(BaseModel):
    prompt: str
    summary: str

@app.post("/generate")
def api_generate(req: PromptRequest):
    result = generate_text(req.prompt, generation_model, generation_tokenizer)
    return {"generated": result}

@app.post("/summarize")
def api_summarize(req: PromptRequest):
    summary = summarize_text(req.prompt, summarizer)
    return {"summary": summary}

@app.post("/similarity")
def api_similarity(req: SimilarityRequest):
    is_relevant, sim_score = check_similarity(req.prompt, req.summary, rouge_metric)
    return {"is_relevant": is_relevant, "similarity": sim_score}

@app.post("/filter")
def api_filter(req: PromptRequest):
    passes, reasons = run_filters(req.prompt, classifier)
    return {"passed": passes, "reasons": reasons}

@app.post("/perplexity")
def api_perplexity(req: PromptRequest):
    ppl = compute_perplexity(req.prompt, generation_model, generation_tokenizer)
    return {"perplexity": ppl}
