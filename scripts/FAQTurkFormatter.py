import csv
import random

file_name = "/home/jdieter/eloquent/Datasets/FAQQuestions.csv"
out_file = "/home/jdieter/eloquent/Datasets/FAQQuestionsTurk"

data = []
with open(file_name, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        data.append(row[0])

random.shuffle(data)
length = len(data)
for i in range(10):
    subdata = data[i*(length//10):min(length, (i+1)*(length//10))]
    with open(out_file + str(i + 1) + ".csv", "w") as o:
        csvwriter = csv.writer(o, quoting=csv.QUOTE_ALL)
        csvwriter.writerow(["question_0", "question_1", "question_2", "question_3"])
        row_out = []
        for row in subdata:
            row_out.append(row)
            if len(row_out) == 4:
                csvwriter.writerow(row_out)
                row_out = []
