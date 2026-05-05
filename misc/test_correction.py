from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def test_model(model_name, prefix, text):
    print(f"\nTesting {model_name} with prefix '{prefix}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(f"{prefix}{text}", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    correction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Output: {correction}")

text = "Amazingly disqualifed"
# test_model("vennify/t5-base-grammar-correction", "grammar: ", text)
# Let's try pszemraj/grammar-synthesis-small (since it's fast)
test_model("pszemraj/grammar-synthesis-small", "", text)
