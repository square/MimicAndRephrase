import csv
import random
rule_based_in = "/home/jdieter/Downloads/xbigdatasettest_baseline.tsv"
gold_in = "/home/jdieter/eloquent/eloquent/python/src/core/idk_rephrasing/model_out.tsv"
out = "/home/jdieter/eloquent/Datasets/abtestturkinrulebased.tsv"

outputs = []

with open(rule_based_in, 'r') as f:
    with open(gold_in, 'r') as f2:
        csvreadergold = csv.reader(f2, delimiter="\t")
        csvreaderrule = csv.reader(f, delimiter="\t")
        for row_rule in csvreaderrule:
            row_gold = next(csvreadergold)
            outputs.append(row_rule[0] + "\t" + row_gold[1] + "\t" + row_rule[1].lower() + "\t" + row_rule[2] + "\n")

with open(out, "w+") as f:
    for out in outputs:
        f.write(out)
