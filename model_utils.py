from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import numpy as np

def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=50256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(text, summarizer):
    summary = summarizer(text, max_length=60, min_length=10, do_sample=False)
    return summary[0]['summary_text']

def check_similarity(prompt, summary, rouge_metric):
    results = rouge_metric.compute(predictions=[summary], references=[prompt])
    score = results["rougeL"].mid.fmeasure
    return score > 0.5, round(score, 2)

def run_filters(text, classifier):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    passed = not (label.lower() == 'toxic' and score > 0.7)
    reasons = []
    if not passed:
        reasons.append(f"Toxic content detected (score: {score:.2f})")
    return passed, reasons

def compute_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()
