import csv

input_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_test.tsv"
output_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_test_condensed.tsv"

out_lines = []

with open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        if line[3] == "condensed":
            out_lines.append(line)

with open(output_file, "w+") as f:
    for line in out_lines:
        f.write("\t".join(line) + "\n")
