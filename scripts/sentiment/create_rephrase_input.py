import csv
import json
input_list = "/home/justin/Eloquent/Datasets/sentiment/scenario_list.tsv"
out_file = "/home/justin/Eloquent/Datasets/sentiment/rephrase_turk_in.jsonl"

positives = []
negatives = []

with open(input_list, 'r') as f:
    csvreader = csv.reader(f, delimiter="\t")
    for row in csvreader:
        if row[1] == "pos":
            positives.append(row[0])
        else:
            negatives.append(row[0])

with open(out_file, "w+") as f:
    while len(positives) >= 6 and len(negatives) >= 6:
        json_object_pos = {"sentiment": "pos", "bonus": 0.75, "estimatedTime": 270, "reward": 0.25}
        pos_slice = positives[-6:]
        positives = positives[:-6]
        json_object_pos["input"] = [{"originalSentence": sent} for sent in pos_slice]
        json_object_neg = {"sentiment": "neg", "bonus": 0.75, "estimatedTime": 270, "reward": 0.25}
        neg_slice = negatives[-6:]
        negatives = negatives[:-6]
        json_object_neg["input"] = [{"originalSentence": sent} for sent in neg_slice]
        f.write(json.dumps(json_object_pos, separators=(',', ':')) + "\n")
        f.write(json.dumps(json_object_neg, separators=(',', ':')) + "\n")