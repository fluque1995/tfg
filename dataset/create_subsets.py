import csv
import random

def create_dict_from_csv(f):
    reader = csv.reader(f)
    return {row[0]: row[1] for row in reader}


def split_in_classes(raw_dict):
    labels_set = set([x for x in raw_dict.values()])
    reverted_dict = {x: [] for x in labels_set}
    for key, val in raw_dict.items():
        reverted_dict[val].append(key)

    return reverted_dict

def create_cross_validation_sets(fname, n_sets=5):
    f = open(fname, 'r')
    records_dict = create_dict_from_csv(f)
    splitted_dict = split_in_classes(records_dict)
    folds_dict = [{} for i in range(n_sets)]
    max_length = 0

    for key, value in splitted_dict.items():
        random.shuffle(value)
        size = len(value)
        i = 0
        for j in range(size):
            folds_dict[i][value[j]] = key
            i += 1
            i %= n_sets

    return folds_dict
