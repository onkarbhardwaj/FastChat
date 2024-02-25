from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ibm/labradorite-13b")
model = AutoModelForCausalLM.from_pretrained("ibm/labradorite-13b")