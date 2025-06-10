from model import ModelKVzip
import torch # Import torch

'''
Please note that a warning from the Hugging Face transformers
library (regarding "The attention mask is not set...") might appear.
This is because KVzip itself isn't forwarding an attention_mask
to the underlying model's generation process.
For short queries like in this example, this warning often doesn't impact
the results, but it's something to be aware of.
Addressing that warning would likely require changes within the KVzip library
itself or using a version that explicitly supports passing attention_mask.

Q: Is it possible to use LiteLLM for inference while keeping KVzip features?
A: Yes.
'''

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
context = "This is my basic profile. My name is Kim living in Seoul. My major is computer science."
queries = ["What is my name?", "Do I live in Seoul?"]

kv = model.prefill(context, load_score=False)  # prefill KV cache + importance scoring
kv.prune(ratio=0.3)  # compression ratio, evict 70% KV

for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(
        query_ids, kv=kv, update_cache=False
    )  # efficient inference
    print(q, output)
