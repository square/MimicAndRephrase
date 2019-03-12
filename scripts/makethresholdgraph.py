import csv
import matplotlib.pyplot as plt
import numpy as np

response_file = "/home/jdieter/Downloads/Batch_3439884_batch_results.csv"

total_processed = -1
num_qs = 5
q_gold_values = []
q_thresh_values = []
answer_indices = []
gold_indices = []
threshold_indices = []
model_preferred = []
with open(response_file, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        if total_processed == -1:

            for i in range(num_qs):
                threshold_indices.append(row.index("Input.Thresh" + str(i + 1)))
                gold_indices.append(row.index("Input.Gold" + str(i + 1)))
                answer_indices.append(row.index("Answer.Q" + str(i + 1) + "Answer"))
            total_processed += 1
            continue
        for i in range(num_qs):
            q_number = num_qs * (total_processed // num_qs) + i
            if len(q_gold_values) <= q_number:
                q_gold_values.append(0)
            if len(q_thresh_values) <= q_number:
                q_thresh_values.append(float(row[threshold_indices[i]]))
            if row[answer_indices[i]] == row[gold_indices[i]]:
                q_gold_values[q_number] += 1
        total_processed += 1
for i in range(len(q_gold_values)):
    model_preferred.append((num_qs - q_gold_values[i])/num_qs)

plt.plot(q_thresh_values, model_preferred, 'ro')
plt.show()
print(np.corrcoef(q_thresh_values, model_preferred))
