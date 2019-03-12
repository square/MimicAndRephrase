import csv
model_out = "/home/jdieter/eloquent/Datasets/abtestturkinrulebased.tsv"
training_data = "/home/jdieter/eloquent/Datasets/xbigdatasettrain.tsv"

training_data_list = []
with open(training_data, 'r') as f:
    csvreader = csv.reader(f, delimiter="\t")
    for row in csvreader:
        training_data_list.extend(row[0].lower().split())

print(training_data_list)
num_words = 0
num_not_in_data = 0
num_rows = 0
num_rows_with_not_in_data = 0
with open(model_out, 'r') as f:
    csvreader = csv.reader(f, delimiter  = '\t')
    for row in csvreader:
        num_rows += 1
        row_data = False
        for word in row[0].lower().split():
            num_words += 1
            if not word in training_data_list:
                num_not_in_data += 1
                if not row_data:
                    num_rows_with_not_in_data += 1
                    row_data = True
print(num_not_in_data / num_words)
print(num_rows_with_not_in_data/num_rows)