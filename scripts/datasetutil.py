import csv
import random

file_name = "/home/jdieter/eloquent/Datasets/longturkclean.tsv"
out_file = "/home/jdieter/eloquent/Datasets/longturkclean2.tsv"

data = []
with open(file_name, 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    for row in csvreader:
        if not (row[1].startswith('I do not know') or row[1].startswith('I am not able to')):
            print(row[0])
            print(row[1])
        else:
            data.append(row)

with open(out_file, "w") as o:
    csvwriter = csv.writer(o, delimiter='\t')
    for row in data:
        csvwriter.writerow(row)