#from huggingface_hub import login
#login()

#-----------
# load pretrained model


from transformers import AutoModelForCausalLM, AutoTokenizer
"""
print("load model")
#model = AutoModelForCausalLM.from_pretrained("arnir0/Tiny-LLM", torch_dtype="auto", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("HuggingfaceTB/Smolvlm-500m-instruct", torch_dtype="auto", device_map="auto")
#model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3-8b-instruct", torch_dtype="auto", device_map="auto")


print("load tokenizer")
#tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3-8b-instruct") # gpu mem crash
#tokenizer = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM") # sentencepiece install issue
tokenizer = AutoTokenizer.from_pretrained("HuggingfaceTB/Smolvlm-500m-instruct") # not made for text generation

print("tokenize input")
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")

print("pass tokenized inputs to generate text")
generated_ids = model.generate(**model_inputs, max_length=30)

print("decode token ids back into text")
print(tokenizer.batch_decode(generated_ids)[0])
#'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
"""


#model_name = "KingNish/Qwen2.5-0.5b-Test-ft" # works, 3 sec inference
model_name = "unsloth/Llama-3-8b-instruct" # works, 20 sec to 5 min inference- is also 10x larger. my hardware setup is limited to run about 0.5b models- only consumes GPU memory but not shared memory... 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# to avoid redownloading weights every new session: model.save_pretrained("./my_model_directory/")  # only needed first run

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Which is greater 9.9 or 9.11 ?? Please apply extended reasoning or Chain-of-Thought in your response."
print(prompt)

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

"""
#-----------
# pipeline

from transformers import pipeline

#pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cuda")
#pipeline = pipeline("text-generation", model="unsloth/Llama-3-8b-instruct", device="cuda")
#pipeline = pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf", device="cuda")
#pipeline = pipeline("text-generation", model="arnir0/Tiny-LLM", device="cuda")
pipeline = pipeline("text-generation", model="HuggingfaceTB/Smolvlm-500m-instruct", device="cuda")

pipeline("The secret to baking a good cake is ", max_length=50)




#-----------
# trainer

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")

def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#trainer.train()
"""
print("test")
