import csv

sample_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_test_small.tsv"
model_out = "/home/justin/Eloquent/Datasets/sentiment/sentiment_model_out_test.tsv"
extra_samples = 5
output_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_model_out_test_small.tsv"

out_lines = []
out_lines_extra = []

sample_qs = []

with open(sample_file) as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        sample_qs.append(" ".join(line[0].split()))

with open(model_out) as f:
    reader = csv.reader(f, delimiter="\t")
    i = 0
    for line in reader:
        if " ".join(line[0].split()) in sample_qs:
            out_lines.append(line)
        elif i < extra_samples:
            out_lines_extra.append(line)
            i += 1

out_lines.extend(out_lines_extra)

with open(output_file, "w+") as f:
    for line in out_lines:
        f.write("\t".join(line) + "\n")
