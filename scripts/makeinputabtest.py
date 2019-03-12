import csv
import random
model_out = "/home/justin/Eloquent/Datasets/sentiment/sentiment_model_out_test_small.tsv"
turk_in = "/home/justin/Eloquent/Datasets/sentiment/sentiment_ab_test_in.tsv"

num_outptus = 400
num_qs = 5

lines = []

num_same = 0
total_processed = 0

def get_n_from_reader(n, reader):
    global num_same
    global total_processed
    nlines = []
    try:
        i = 0
        while i < n:
            row = next(reader)
            if row[1] != row[2]:
                nlines.append(row)
                i += 1
            else:
                num_same += 1
            total_processed += 1
        return nlines
    except StopIteration:
        return None

with open(model_out, 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    outputs_printed = 0
    n_lines = get_n_from_reader(num_qs, csvreader)
    finished_outputs = False
    while n_lines != None:
        row_string = ""
        for i in range(num_qs):
            line = n_lines[i]
            if random.uniform(0, 1) < .5:
                row_string = row_string + line[0] + "\t" + line[1] + "\t" + line[2] + "\tA\t" + line[3] + "\t"
            else:
                row_string = row_string + line[0] + "\t" + line[2] + "\t" + line[1] + "\tB\t" + line[3] + "\t"
        lines.append(row_string)
        n_lines = get_n_from_reader(num_qs, csvreader)
        if num_outptus <= total_processed and not finished_outputs:
            print("Total processed: " + str(total_processed))
            print("Num same: " + str(num_same))
            print("Last question: " + str(n_lines[-1][0]))
            finished_outputs = True

    print("Total processed: " + str(total_processed))
    print("Num same: " + str(num_same))

header = "Q1\tQ1A\tQ1B\tGold1\tThresh1\tQ2\tQ2A\tQ2B\tGold2\tThresh2\tQ3\tQ3A\tQ3B\tGold3\tThresh3\tQ4\tQ4A\tQ4B\tGold4\tThresh4\tQ5\tQ5A\tQ5B\tGold5\tThresh5\t"
with open(turk_in, "w+") as f:
    f.write(header + "\n")
    for line in lines:
        f.write(line + "\n")
