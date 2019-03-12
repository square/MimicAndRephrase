import csv

dataset_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_cleaned.tsv"
output_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset_cleaned2.tsv"

lines = []

with open(dataset_file) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        lines.append(row)


for row in lines:
    if len(row[1].split()) >= 2 and ' '.join(row[1].split()[0:2]).lower() == "i 'm":
        row[1] = "I am " + ' '.join(row[1].split()[2:])
    if len(row[1].split()) >= 1 and row[1].split()[0].lower() == "sorry":
        row[1] = "I am sorry " + ' '.join(row[1].split()[1:])
    if len(row[1].split()) >= 1 and (row[1].split()[0].lower() == "good" or row[1].split()[0].lower() == "glad"
                                     or row[1].lower().startswith("i am so happy")
                                     or row[1].lower().startswith("i am so glad")):
        row[1] = "I am glad " + ' '.join(row[1].split()[1:])

    if row[1].lower().startswith("i am so sorry"):
            row[1] = "I am sorry " + ' '.join(row[1].split()[1:])

i = 0
out_lines = []
for row in lines:
    if row[1].lower().startswith("i am sorry") or row[1].lower().startswith("i am glad") \
            or row[1].lower().startswith("i am happy") or row[1].lower().startswith("i am sad"):
        out_lines.append(row)
        i += 1

print(i)

with open(output_file, "w+") as f:
    for line in out_lines:
        f.write("\t".join(line) + "\n")
