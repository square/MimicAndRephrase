import random
num_samples = 400

input_file = "/home/justin/Eloquent/Datasets/idk/idkdatasettest.tsv"
output_file = "/home/justin/Eloquent/Datasets/idk/idkdatasettest_small.tsv"

in_lines = []
with open(input_file, "r") as f:
    in_lines = f.readlines()

with open(output_file, "w+") as f:
    for line in in_lines[0:num_samples]:
        f.write(line)
