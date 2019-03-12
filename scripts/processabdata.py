import csv

response_file = "/home/justin/Downloads/Batch_3552822_batch_results.csv"

gold_preferred = 0
total_processed = -1
gold_indices = []
answer_indices = []
num_qs = 5
with open(response_file, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        if total_processed == -1:
            for i in range(num_qs):
                gold_indices.append(row.index("Input.Gold" + str(i + 1)))
                answer_indices.append(row.index("Answer.Q" + str(i + 1) + "Answer"))
            total_processed += 1
            continue
        for i in range(num_qs):
            if total_processed >= 1275 and i >= 1 :
                break
            if row[gold_indices[i]] == row[answer_indices[i]]:
                gold_preferred += 1
            total_processed += 1
print("Processed total of " + str(total_processed) + " AB tests")
print("The gold value was preferred " + str(gold_preferred) + " times")
print("The model value was preferred " + str(total_processed - gold_preferred) + " times")
print("The gold was chosen with prob " + str(gold_preferred / total_processed))
print("The models value was chosen with prob " + str((total_processed - gold_preferred) / total_processed))
