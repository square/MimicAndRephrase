import csv
import math
from random import shuffle
input_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_cleaned2.tsv"
train_out = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_train.tsv"
dev_out = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_dev.tsv"
test_out = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_test.tsv"

# Examples that have both condensed and full response
examples_with_both = []
# Examples that only have one
scenarios_with_singular = []
examples_with_singular = []

with open(input_file, 'r') as f:
    csvreader = csv.reader(f, delimiter ="\t")
    for row in csvreader:
        original_sentence = row[0]
        if original_sentence not in scenarios_with_singular:
            last_original_sentence = original_sentence
            scenarios_with_singular.append(original_sentence)
            examples_with_singular.append([row])
        else:
            i = scenarios_with_singular.index(original_sentence)
            del scenarios_with_singular[i]
            example = examples_with_singular[i].copy()
            del examples_with_singular[i]
            example.append(row)
            examples_with_both.append(example)

test_examples = []
train_examples = []
dev_examples = []
for example_source in [examples_with_both, examples_with_singular]:
    pos_examples = []
    neg_examples = []
    for example in example_source:
        if example[0][2] == "pos":
            pos_examples.append(example)
        else:
            neg_examples.append(example)
    shuffle(pos_examples)
    shuffle(neg_examples)
    for data in [pos_examples, neg_examples]:
        a2 = math.floor(len(data) * .15)
        a3 = math.floor(len(data) * .30)
        dev_examples.extend(data[:a2])
        test_examples.extend(data[a2:a3])
        train_examples.extend(data[a3:])

shuffle(test_examples)
shuffle(dev_examples)
shuffle(train_examples)

with open(test_out, "w+") as f:
    for example in test_examples:
        for row in example:
            f.write("\t".join(row) + "\n")
with open(dev_out, "w+") as f:
    for example in dev_examples:
        for row in example:
            f.write("\t".join(row) + "\n")
with open(train_out, "w+") as f:
    for example in train_examples:
        for row in example:
            f.write("\t".join(row) + "\n")

