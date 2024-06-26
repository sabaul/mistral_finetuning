import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
custom_trained_model = "mistralai-medical"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={'': 0}
)
model = PeftModel.from_pretrained(model, custom_trained_model)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda:0"

def stream(prompt, step):
    messages = [
        {"role": "user", "content": f"Given the input: {prompt}\n\n{step}\n\n"}
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]


def entire_process(prompt):
    step1 = "Extract medical terms from the input above and number them line by line:\n"
    step1_output = stream(prompt, step1)
    print(step1_output)
    print('\n'*3)
    
    input_sentence = input("Enter the context you want to find:\n")
    step2 = f"Extract set of sentences which have the same context as: {input_sentence}"
    step2_output = stream(prompt, step2)
    print(step2_output.replace("<s> [INST] Given the input:", '').replace(prompt, ''))
    print('\n'*3)
    
    step3 = "Summarize the content above:\n"
    step3_output = stream(prompt, step3)
    print(step3_output.replace("<s> [INST] Given the input:", '').replace(prompt, ''))
    print('\n'*3)

if __name__ == "__main__":
    with open('sample_input.txt', 'r') as f:
        prompt = f.read()
    entire_process(prompt)