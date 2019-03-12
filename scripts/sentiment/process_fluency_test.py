import json

input_files = ["/home/justin/Eloquent/eloquent/turk/experiments/RateFluency.1-20190302/.bk-0/outputs.jsonl"]

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
fluent = 0
semifluent = 0
nonfluent = 0

for annotation in annotations:
    for individual_annotation in annotation:
        for response in individual_annotation:
            total += 1
            if response == -1:
                fluent += 1
            if response == 1:
                semifluent += 1
            if response == 2:
                nonfluent += 1

print(total)
print(fluent)
print(fluent/total)
print(semifluent)
print(semifluent/total)
print(nonfluent)
print(nonfluent/total)