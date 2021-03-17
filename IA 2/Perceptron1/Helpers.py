import csv
import numpy as np


def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        info = []
        for row in reader:
            info.append(row)

        for datum in info:
            for i in range(len(info[0])):
                datum[i] = float(datum[i])

        info = np.vstack((info))
        return info
