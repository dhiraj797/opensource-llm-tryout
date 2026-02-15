from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading model... (this may take a few minutes the first time)")

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Model loaded!\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Llama: {response}\n")
