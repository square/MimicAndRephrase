import csv
import random

file_name = "/home/jdieter/eloquent/Datasets/xbigdataset.tsv"
out_file_train = "/home/jdieter/eloquent/Datasets/xbigdatasettrain.tsv"
out_file_dev = "/home/jdieter/eloquent/Datasets/xbigdatasetdev.tsv"
out_file_test = "/home/jdieter/eloquent/Datasets/xbigdatasettest.tsv"

data = []
with open(file_name, 'r') as f:
    data = [line.rstrip('\n') for line in f]

random.shuffle(data)
length = len(data)

with open(out_file_dev, "w") as f:
    f.writelines(["%s\n" % item for item in data[:3*len(data)//20]])

with open(out_file_test, "w") as f:
    f.writelines(["%s\n" % item for item in data[3*len(data)//20:6*len(data)//20]])

with open(out_file_train, "w") as f:
    f.writelines(["%s\n" % item for item in data[6*len(data)//20:]])


