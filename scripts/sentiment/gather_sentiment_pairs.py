import json
input_files = ["/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.3-20190209/inputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.4-20190209/inputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.6-20190210/inputs.jsonl",
               "/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.7-20190211/inputs.jsonl"]
output_files = ["/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.3-20190209/outputs.jsonl",
                "/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.4-20190209/.bk-0/outputs.jsonl",
                "/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.6-20190210/.bk-0/outputs.jsonl",
                "/home/justin/Eloquent/eloquent/turk/experiments/ScenarioRephrase.7-20190211/.bk-0/outputs.jsonl"]
out_file = "/home/justin/Eloquent/Datasets/sentiment/sentiment_dataset.tsv"

scenarios = []
for i in range(len(input_files)):
    with open(input_files[i]) as sentiment_input, open(output_files[i]) as output:
        line = output.readline()
        sentiment_line = sentiment_input.readline()
        while line:
            sentiment_data = json.loads(sentiment_line)
            sentiment = sentiment_data["sentiment"]
            data = json.loads(line)
            for key in data.keys():
                for scenario in data[key]["Answer"]["responses"]:
                    originalSentence = scenario["originalSentence"]
                    if len(scenario["condensed"]["responses"]) > 0:
                        condensedSentence = scenario["condensed"]["responses"][0]
                        scenarios.append((originalSentence, condensedSentence, sentiment, "condensed"))
                    if len(scenario["full"]["responses"]) > 0:
                        fullSentence = scenario["full"]["responses"][0]
                        scenarios.append((originalSentence, fullSentence, sentiment, "full"))
            line = output.readline()
            sentiment_line = sentiment_input.readline()

with open(out_file, "w+") as f:
    for scenario in scenarios:
        f.write(scenario[0] + "\t" + scenario[1] + "\t" + scenario[2] + "\t" + scenario[3] + "\n")