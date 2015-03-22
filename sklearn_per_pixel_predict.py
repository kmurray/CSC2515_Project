#!/usr/bin/env python
import argparse
import os
import os.path as path
import csv
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import svm, neighbors, tree
from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Convert png data to csv")

    parser.add_argument("-t", "--train_data", 
                        required=True,
                        help="CSV of training data")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    print "Loading data: ", args.train_data
    X = None #Data
    y = None #Labels
    
    with open(args.train_data) as f:
        print "Cols: ", f.readline()
        data = np.loadtxt(f, delimiter=",")

        X = data[:, 1:]
        y = data[:, 0]

    print "Splitting data into train, test validation"
    train_validation_data, test_data, train_validation_targets, test_targets = train_test_split(X, y, test_size=0.3)

    train_data, validation_data, train_targets, validation_targets = train_test_split(train_validation_data, train_validation_targets, test_size=10./70.)

    print "Training Classifier"
    #classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_data, train_targets)

    #Evaluation
    print "Evaluating Classifier"
    test_targets_pred = classifier.predict(test_data)
    print metrics.classification_report(test_targets, test_targets_pred)

if __name__ == "__main__":
    main()
