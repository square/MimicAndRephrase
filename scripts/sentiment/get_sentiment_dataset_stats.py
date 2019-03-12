import csv

dataset_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset.tsv"
condensed_differences = []
examples = []
with open(dataset_file) as f:
    csvreader = csv.reader(f, delimiter="\t")
    prev_original = ""
    prev_response_length = 0
    for row in csvreader:
        original = row[0]
        response = row[1]
        positive = row[2]
        condensed = row[3]
        original_length = len(original.split())
        response_length = len(response.split())
        examples.append((row, original_length, response_length))
        if prev_original == original:
            condensed_differences.append((original, abs(response_length - prev_response_length)))
        prev_original = original
        prev_response_length = response_length

sorted_list = sorted(condensed_differences, key=lambda x: x[1])
avg = 0
for diff in sorted_list:
    avg += diff[1]
avg /= len(sorted_list)
print(avg)
sorted_examples = sorted(examples, key=lambda x: -abs(x[1] - x[2]))
avg_response_diff = 0
for example in sorted_examples:
    avg_response_diff += abs(example[1] - example[2])
avg_response_diff /= len(sorted_examples)
print(avg_response_diff)