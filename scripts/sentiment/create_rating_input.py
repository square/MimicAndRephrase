import csv
import json
import random

input_file = "/home/justin/Eloquent/Datasets/idk/idkdatasettest_small.tsv"
output_file = "/home/justin/Eloquent/Datasets/idk/idk_dataset_fluency_turk_in.jsonl"

#task = "sentiment"
#taskVerb = "understood what emotional sentiment was conveyed by"
#task = "Respond"
#taskVerb = "understood what was asked by "
bonus = 0.2
reward = 0.24
estimatedTime = 60

responses = []
with open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    i = 0
    for line in reader:
        #responses.append({"id": i, "prompt": line[0], "response": line[1]})
        responses.append({"id": i, "value": line[1]})
        i += 1

random.shuffle(responses)
with open(output_file, "w+") as f:
    for i in range(len(responses)//16):
        inputs = responses[16*i: 16*i+16].copy()
        #input_line = {"inputs": inputs, "task": task, "taskVerb": taskVerb}
        json_object = {"input": inputs, "bonus": bonus, "reward": reward, "estimatedTime": estimatedTime}
        f.write(json.dumps(json_object, separators=(',', ':')) + "\n")
