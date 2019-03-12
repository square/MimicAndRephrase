import csv
from core.nn.token_mapper import RegexTokenMapping, ExactTokenMapping, TokenMapper, HashTokenMapping
from core.utils.core_nlp import SimpleSentence
from core.utils.data import NearNeighborLookup
from core.vectorspace.word_embedding import Glove
import json
import random

dataset_file = "/home/justin/Eloquent/Datasets/idk/idkdataset.tsv"
input_file = "/home/justin/Eloquent/Datasets/idk/idkdatasettest_small.tsv"
output_file = "/home/justin/Eloquent/Datasets/idk/idk_dataset_specificity_test_in.jsonl"

bonus = 0.2
reward = 0.20
estimated_time = 75
datapoints_per_batch = 10

token_mappings = [
    RegexTokenMapping("^[0-9]$", 3, "NUM_1"),
    RegexTokenMapping("^[0-9]{2}$", 3, "NUM_2"),
    RegexTokenMapping("^[0-9]{3}$", 3, "NUM_3"),
    RegexTokenMapping("^[0-9]+$", 3, "NUM_MANY"),
    RegexTokenMapping("^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", 3, "EMAIL"),
    ExactTokenMapping(["<SOS>", "<EOS>"]),
]

print('Adding {} token mappings'.format(len(token_mappings)))
glove = Glove.from_binary().with_new_token_mapper(
    token_mapper=TokenMapper(token_mappings, [HashTokenMapping(10)])
)

question_list_pos = []
question_list_neg = []

do_sentiment = False

with open(dataset_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        if (not do_sentiment) or line[2] == "pos":
            question_list_pos.append(line[0])
        else:
            question_list_neg.append(line[0])
sslist_pos = [SimpleSentence.from_text(sentence) for sentence in question_list_pos]
nns_pos = NearNeighborLookup.from_sentences(sslist_pos, glove)
if do_sentiment:
    sslist_neg = [SimpleSentence.from_text(sentence) for sentence in question_list_neg]
    nns_neg = NearNeighborLookup.from_sentences(sslist_neg, glove)

response_sets = []

with open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        if (not do_sentiment) or line[2] == "pos":
            neighbors = nns_pos.find_neighbors(SimpleSentence.from_text(line[0]), 2, ignore_self=True)
            sentiment = "pos"
        else:
            neighbors = nns_neg.find_neighbors(SimpleSentence.from_text(line[0]), 2, ignore_self=True)
            sentiment = "neg"
        neighbors = [" ".join(entry.original_texts()) for entry in neighbors]

        response_sets.append((line[0], neighbors, line[1], sentiment))

turk_inputs = []
i = 0
for response in response_sets:
    prompts = response[1].copy()
    prompts.append(response[0])
    random.shuffle(prompts)
    json_object = {
        "id": i,
        "prompts": prompts,
        "response": response[2],
        "correct": prompts.index(response[0]),
        "sentiment": response[3]
    }
    turk_inputs.append(json_object)
    i += 1

random.shuffle(turk_inputs)

with open(output_file, "w+") as f:
    for i in range(len(response_sets)//datapoints_per_batch):
        turk_input = turk_inputs[datapoints_per_batch * i: datapoints_per_batch * (i + 1)]
        json_object = {
            "input": turk_input,
            "bonus": bonus,
            "reward": reward,
            "estimatedTime": estimated_time
        }
        f.write(json.dumps(json_object) + "\n")


