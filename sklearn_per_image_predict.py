#!/usr/bin/env python
import argparse
import os
import os.path as path
import csv
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import Image

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import svm, neighbors, tree
from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Convert png data to csv")

    parser.add_argument("-d", "--data_dir", 
                        required=True,
                        help="KITTI data directory")

    parser.add_argument("-t", "--road_type",
                        choices=["um", "uu", "umm", "all"],
                        default="um",
                        help="Road types to consider. Default: %(default)s")

    parser.add_argument("-c", "--classification_type",
                        choices=["lane", "road"],
                        default="lane",
                        help="Type of classification. Default: %(default)s")

    parser.add_argument("-n", "--num_images",
                    type=int,
                    help="Override the number of images to process. Default all.")

    parser.add_argument("--width",
                        type=int,
                        default=1242,
                        help="Width of images. Default: %(default)s (will be resized if different)")
    parser.add_argument("--height",
                        type=int,
                        default=375,
                        help="Height of images. Default: %(default)s (will be resized if different)")
    parser.add_argument('--channels',
                        type=int,
                        default=3,
                        help="Number of colour channels. Default %(default)s")

    args = parser.parse_args()

    assert args.channels == 3

    return args

def main():
    args = parse_args()

    print "Loading data from: ", args.data_dir
    X, y = load_data(args)

    #Save output
    np.save("X.npy", X)
    np.save("y.npy", y)

    print "Splitting data into train, test, validation"
    train_validation_data, test_data, train_validation_targets, test_targets = train_test_split(X, y, test_size=0.3, random_state=0)

    train_data, validation_data, train_targets, validation_targets = train_test_split(train_validation_data, train_validation_targets, test_size=10./70., random_state=0)
    print "\tTrain %d, Validate %d, Test %d" % (train_data.shape[0], validation_data.shape[0], test_data.shape[0])

    print "Training Classifier"
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3)

    classifier.fit(train_data, train_targets)

    #Evaluation
    print "Evaluating Classifier"
    test_targets_pred = classifier.predict(test_data)

    #Difference of test and actual
    err_frac = np.sum(np.abs(test_targets - test_targets_pred)) / np.prod(test_targets.shape)
    print "Error Rate: %f" % err_frac

    plot_classification(args, classifier, test_data, test_targets, test_targets_pred)

def load_data(args):
    #Collect image paths
    train_dir = path.join(args.data_dir, "training", "image_2")
    train_images = []
    for img in os.listdir(train_dir):
        if img.startswith(args.road_type) or args.road_type == "all":
            train_images.append(path.join(train_dir,img))
    train_images.sort()

    if not args.num_images:
        args.num_images = len(train_images)

    gt_dir = path.join(args.data_dir, "training", "gt_image_2")
    gt_images = []
    for img in os.listdir(gt_dir):
        if img.startswith(args.road_type) or args.road_type == "all":
            if args.classification_type in img:
                gt_images.append(path.join(gt_dir, img))
    gt_images.sort()

    #Load the data
    # For X (input data) we expect args.num_images images of dimension args.w*args.h with args.channels colour channels.
    # As a result we expect the X training data to have args.num_images rows (one for each image) and
    # args.w*args.h*args.channels columns (one for each feature = pixelcolor)
    #
    # For Y (targets) we expect args.num_images images fo dimension args.w*args.h with a single colour channel
    # As a result we expect Y to have args.num_images rows and args.w*args.h columns
    npixels = args.width*args.height
    X = np.empty([args.num_images, npixels*args.channels])
    y = np.empty([args.num_images, npixels])
    for i in xrange(len(train_images)):
        train_image = Image.open(train_images[i])
        gt_image = Image.open(gt_images[i])

        #Resize
        if train_image.size != (args.width, args.height):
            print "Resizing %s from %s to %s" % (train_images[i], train_image.size, (args.width, args.height))
            train_image = train_image.resize((args.width,args.height), Image.ANTIALIAS)
        if gt_image.size != (args.width, args.height):
            print "Resizing %s from %s to %s" % (gt_images[i], gt_image.size, (args.width, args.height))
            gt_image = gt_image.resize((args.width,args.height), Image.ANTIALIAS)

        train_pixels = np.array(train_image) / 255.0
        gt_pixels = np.array(gt_image) / 255.0

        #Reshape
        #X[i,0:npixels] = train_pixels[:,:,0].flatten()
        #X[i,npixels:2*npixels] = train_pixels[:,:,1].flatten()
        #X[i,2*npixels:] = train_pixels[:,:,2].flatten()
        X[i,:] = train_pixels.reshape([npixels*args.channels])

        y[i] = gt_pixels[:,:,2].flatten()



        if args.num_images and i + 1 >= args.num_images:
            break

    return X, y

def plot_classification(args, classifier, test_data, test_targets, predicted_targets):

    img_flat = test_data[0,:]
    img = img_flat.reshape([args.height,args.width,3])

    gt_flat = test_targets[0,:]
    gt = gt_flat.reshape([args.height,args.width])

    pred_flat = predicted_targets[0,:]
    pred = pred_flat.reshape([args.height,args.width])

    fig, axarr = plt.subplots(2, 2)

    axarr[0][0].imshow(img)
    axarr[0][0].set_title("Test Image")

    axarr[1][0].set_title("GroundTruth - Prediction Error")
    axarr[1][0].imshow(np.abs(gt - pred))

    axarr[0][1].set_title("Ground Truth")
    axarr[0][1].imshow(gt)

    axarr[1][1].set_title("Prediction")
    axarr[1][1].imshow(pred)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
