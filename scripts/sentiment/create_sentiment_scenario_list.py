import json
input_files = {"/home/justin/Eloquent/eloquent/turk/experiments/SentimentScenario.4-20190205/.bk-0/outputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/SentimentScenario.3-20190201/.bk-0/outputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/SentimentScenario.6-20190207/.bk-0/outputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/SentimentScenario.7-20190207/.bk-0/outputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/SentimentScenario.8-20190208/.bk-0/outputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/tian/outputs2.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/tian/outputs1.jsonl"}
out_file = "/home/justin/Eloquent/Datasets/sentiment/scenario_list.tsv"

scenarios = []
for file in input_files:
    with open(file) as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            for key in data.keys():
                for scenario in data[key]["Answer"]["responses"][0]["negativeOutput"]["responses"]:
                    scenarios.append((scenario, 0))
                for scenario in data[key]["Answer"]["responses"][0]["positiveOutput"]["responses"]:
                    scenarios.append((scenario, 1))
            line = f.readline()

with open(out_file, "w+") as f:
    for scenario in scenarios:
        f.write(scenario[0] + "\t")
        if scenario[1] == 1:
            f.write("pos\n")
        else:
            f.write("neg\n")
