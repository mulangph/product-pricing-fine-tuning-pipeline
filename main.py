import os
import re
import json
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from pricer.items  import Item
from pricer.evaluator import evaluate


load_dotenv(override=True)
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

username = "mulangph"
dataset = f"mulangph/items_lite"

MODEL_NAME="gpt-4.1-nano"

#====================== Load Training Dataset from Huggingface ======================#
train, val, test = Item.from_hub(dataset)
print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")

openai = OpenAI()

# OpenAI recommends fine-tuning with populations of 50-100 examples
# Running 100 examples (and 1 epoch)
fine_tune_train = train[:100]
fine_tune_validation = val[:50]

#====================== Prepare the data for fine-tuning in JSONL format and upload to OPENAI ======================#
def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"${item.price:.2f}"}
    ]

# Convert the items into a list of json objects - a "jsonl" string
# Each row represents a message in the form:
# {"messages" : [{"role": "system", "content": "You estimate prices...
def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()

# Convert the items into jsonl and write them to a file
def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

write_jsonl(fine_tune_train, "jsonl/fine_tune_train.jsonl")

write_jsonl(fine_tune_validation, "jsonl/fine_tune_validation.jsonl")

with open("jsonl/fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")

with open("jsonl/fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")


#====================== Fine-Tuning Part ======================#
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model=MODEL_NAME,
    seed=42,
    hyperparameters={"n_epochs": 1, "batch_size": 1},
    suffix="pricer"
)

openai.fine_tuning.jobs.list(limit=1)

job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id

openai.fine_tuning.jobs.retrieve(job_id)

openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data


#====================== Test the fine tuned model ======================#

fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model

# The prompt
def test_messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [
        {"role": "user", "content": message},
    ]

test_messages_for(test[0])

# The inference function
def gpt_4__1_nano_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name,
        messages=test_messages_for(item),
        max_tokens=7
    )
    return response.choices[0].message.content

print(test[0].price)
print(gpt_4__1_nano_fine_tuned(test[0]))