#!/usr/bin/env python
import argparse
import numpy as np
import tempfile
from fann2 import libfann

num_neurons_hidden = 20;
max_epochs = 12;
desired_error = 0.001;
learning_rate = 0.1

def parse_args():
    parser = argparse.ArgumentParser(description="Convert png data to csv")

    parser.add_argument("-f", "--fann_file",
                        required=True,
                        help="Fann to load")

    parser.add_argument('-p', '--pca',
                        required=True,
                        help="PCA data to load")

    parser.add_argument('-t', '--type',
                        required=True,
                        help="Road or Lane and uu, um, umm, all")

    args = parser.parse_args()

    return args

def XY_to_fann_train_data(X, Y):
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of lines.")

    train_data = libfann.training_data()

    if len(X):
        dim_X, dim_Y = len(X[0]), len(Y[0])

        tmp = tempfile.NamedTemporaryFile(delete=False)
        with tmp:
            tmp.write("%d %d %d\n"%(len(X), dim_X,  dim_Y))
            for i in xrange(len(X)):
                for line in [ X[i], Y[i] ]:
                    tmp.write("%s\n"% ' '.join( str(float(val)) for val in line ))

        train_data.read_train_from_file(tmp.name)
        tmp.unlink(tmp.name)

    return train_data

def main():
    args = parse_args()

    print "Loading data"
    
    x_train = np.load("data/x-train-pca%s_%s.npy" % (args.pca, args.type))
    x_valid = np.load("data/x-valid-pca%s_%s.npy" % (args.pca, args.type))
    y_train = np.load("data/y-train_%s.npy" % args.type)
    y_valid = np.load("data/y-valid_%s.npy" % args.type)

    print "x_train Shape", x_train.shape
    print "x_valid Shape", x_valid.shape
    print "y_train Shape", y_train.shape
    print "y_valid Shape", y_valid.shape

    print "Creating ann"

    ann = libfann.neural_net()
    ann.create_standard_array((x_train.shape[1], num_neurons_hidden, y_train.shape[1]))
    ann.set_bit_fail_limit(0.5)
    ann.set_learning_rate(learning_rate)

    print "Converting training data"
    train_data = XY_to_fann_train_data(x_train, y_train)
    print "Converting testing data"
    test_data = XY_to_fann_train_data(x_valid, y_valid)
    
    ann.train_on_data(train_data, max_epochs, 1, desired_error)

    ann.test_data(train_data)
    bit_fail = ann.get_bit_fail()
    train_err = (float(bit_fail) / (y_train.shape[0] * y_train.shape[1]))

    print "Train Error:", train_err

    ann.test_data(test_data)
    bit_fail = ann.get_bit_fail()
    valid_err = (float(bit_fail) / (y_valid.shape[0] * y_valid.shape[1]))

    print "Validation Error:", valid_err

    print "Saving ANN to %s" % args.fann_file
    ann.save("nnets/%s" % args.fann_file)

if __name__ == "__main__":
    main()
