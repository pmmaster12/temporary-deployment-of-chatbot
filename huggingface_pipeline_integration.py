from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
def model():
 model_id = "meta-llama/Llama-3.2-1B"
 tokenizer = AutoTokenizer.from_pretrained(model_id)
 model = AutoModelForCausalLM.from_pretrained(model_id)
 pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=5000)
 hf = HuggingFacePipeline(pipeline=pipe)
 return hf
 