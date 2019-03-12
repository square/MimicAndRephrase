import csv

from nltk.translate.bleu_score import sentence_bleu

response_file = "/home/jdieter/Downloads/out.tsv"
dev_set_file = "/home/jdieter/eloquent/Datasets/xbigdatasetdev.tsv"

correct = []
data = []
with open(response_file, 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    for row in csvreader:
        if row[0] == "EMPTY":
            data.append(("I do not know how to help with ' " + row[1] + " ' ").split())
        else:
            data.append(row[0].split())
        correct.append(row[2].split())
data = data[:-1]

print(data)
print(len(data))

'''
correct = []
with open(dev_set_file, 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    for row in csvreader:
        correct.append(row[1].split())
'''

print(correct)
print(len(correct))

samples = 0
total_bleu = 0
for i in range(len(data)):
    prediction = [token.lower() for token in data[i]]
    correct_tokens = [token.lower() for token in correct[i]]
    total_bleu += sentence_bleu([correct_tokens], prediction)
    samples += 1

print("Bleu: " + str(total_bleu/samples))