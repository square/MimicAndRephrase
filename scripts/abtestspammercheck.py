import csv

response_file = "/home/jdieter/Downloads/Batch_3439884_batch_results.csv"

workers = {}
worker_column = -1
total_processed = -1
num_qs = 5
q_a_values = []
worker_response = {}
answer_indices = []
with open(response_file, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        if total_processed == -1:
            worker_column = row.index("WorkerId")
            for i in range(num_qs):
                answer_indices.append(row.index("Answer.Q" + str(i + 1) + "Answer"))
            total_processed += 1
            continue
        for i in range(num_qs):
            q_number = 5 * (total_processed // 5) + i
            if len(q_a_values) <= q_number:
                q_a_values.append(0)
            if row[answer_indices[i]] == "A":
                q_a_values[q_number] += 1
            if not row[worker_column] in worker_response:
                worker_response[row[worker_column]] = {}
            worker_response[row[worker_column]][q_number] = row[answer_indices[i]]
        if row[worker_column] in workers:
            workers[row[worker_column]] += 1
        else:
            workers[row[worker_column]] = 1
        total_processed += 1
print("Processed " + str(total_processed) + " hits.")
sorted_turkers = sorted(workers, key=workers.get)
for turker in sorted_turkers:
    amount_disagreed = 0
    for q_number in worker_response[turker]:
        num_a = q_a_values[q_number]
        if worker_response[turker][q_number] == "A":
            amount_disagreed += 5 - num_a
        if worker_response[turker][q_number] == "B":
            amount_disagreed += num_a
    percent_disagreed = amount_disagreed / (4*5*workers[turker])
    print("Worker " + str(turker) + " worked on " + str(workers[turker]) + " HITS and disagreed with other Turkers " + str(percent_disagreed) + " percent of the time.")
