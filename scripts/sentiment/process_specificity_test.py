import json

input_files = ["/home/justin/Eloquent/eloquent/turk/experiments/RateSpecificity.1-20190301/.bk-0/outputs.jsonl"]

hit_reference_file = "/home/justin/Eloquent/eloquent/turk/experiments/RateSpecificity.1-20190301/inputs.jsonl"

golds = []

with open(hit_reference_file) as f:
    line = f.readline()
    while line:
        data = json.loads(line)
        golds.append([datum["correct"] for datum in data["input"]])
        line = f.readline()

annotations = []

for file in input_files:
    with open(file) as f:
        line = f.readline()
        i = 0
        while line:
            data = json.loads(line)
            for key in data.keys():
                annotation = data[key]["Answer"]["responses"]
                if len(annotations) <= i:
                    annotations.append([annotation])
                else:
                    annotations[i].append(annotation)
            i += 1
            line = f.readline()

total = 0
correct = 0
wrong_answer = 0
multiple_none = 0

for i in range(len(annotations)):
    annotation = annotations[i]
    for individual_annotation in annotation:
        for j in range(len(individual_annotation)):
            response = individual_annotation[j]
            total += 1
            if response == 3:
                multiple_none += 1
            elif response == golds[i][j]:
                correct += 1
            else:
                wrong_answer += 1

print(total)
print(correct)
print(correct/total)
print(multiple_none)
print(multiple_none/total)
print(wrong_answer)
print(wrong_answer/total)
