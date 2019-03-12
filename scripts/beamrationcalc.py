import numpy as np
import csv

file_name = "/home/jdieter/eloquent/Datasets/xbigdatasetdev.tsv"

q_l = []
a_l = []
with open(file_name, 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    for row in csvreader:
        q_l.append(len(row[0].split()))
        a_l.append(len(row[1].split()))

q_l = np.array(q_l)
a_l = np.array(a_l)

print(str(a_l/q_l))

print("Ratio: " + str(np.mean(a_l/q_l)))
print("Var: " + str(np.var(a_l/q_l)))