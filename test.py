import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

# Data preprocessing
# coqa = pd.read_json('coqa-train-v1.0.json')
# print(coqa.head())

# del coqa["version"]

# cols = ["text", "question", "answer"]
# comp_list = []
# for index, row in coqa.iterrows():
#     for i in range(len(row["data"]["questions"])):
#         temp_list = []
#         temp_list.append(row["data"]["story"])
#         temp_list.append(row["data"]["questions"][i]["input_text"])
#         temp_list.append(row["data"]["answers"][i]["input_text"])
#         comp_list.append(temp_list)

# new_def = pd.DataFrame(comp_list, columns=cols)
# new_def.to_csv("CoQA_data.csv", index=False)

# End of data preprocessing

data = pd.read_csv("CoQA_data.csv")
print(data.head())

# Model Download and initialization
print("Beginning downloading and initialization of pretrained model")

BERT_MODEL = 'bert-large-uncased-whole-word-masking-finetuned-squad'

model = BertForQuestionAnswering.from_pretrained(BERT_MODEL)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

print("Pretrain model initialization done")

# test
random_num = np.random.randint(0, len(data))
question = data["question"][random_num]
text = data["text"][random_num]

input_ids = tokenizer.encode(question, text)
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(input_ids)
print(tokens)
sep_idx = input_ids.index(tokenizer.sep_token_id)
num_seg_a = sep_idx+1
num_seg_b = len(input_ids) - num_seg_a
segment_ids = [0]*num_seg_a + [1]*num_seg_b

print(segment_ids)

output = model(torch.tensor([input_ids]))
token_type_ids = torch.tensor([segment_ids])

answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)

if answer_end >= answer_start:
    answer = tokens[answer_start]
    for i in range(answer_start+1, answer_end+1):
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]

if answer.startswith("[CLS]"):
    answer = "Ga ada jawaban untuk itu"

print("\nQuestion: {}".format(question.capitalize()))
print("\nPredicted answer:{}".format(answer.capitalize()))