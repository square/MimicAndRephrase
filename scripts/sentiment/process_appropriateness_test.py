import json

input_files = ["/home/justin/Eloquent/eloquent/turk/experiments/RateAppropriateness.3-20190302/.bk-0/outputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/RateAppropriateness.2-20190302/.bk-0/outputs.jsonl"]

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
appropriate = 0

for annotation in annotations:
    for individual_annotation in annotation:
        for response in individual_annotation:
            total += 1
            if response == -1:
                appropriate += 1

print(total)
print(appropriate)
print(appropriate/total)