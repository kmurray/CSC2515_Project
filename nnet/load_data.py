#!/usr/bin/env python
import argparse
import os
import os.path as path
import csv
import numpy as np
import logging
import matplotlib.pyplot as plt
from time import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import Image

from sklearn import preprocessing, decomposition
from sklearn.cross_validation import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Convert png data to csv")

    parser.add_argument("-d", "--data_dir",
                        required=True,
                        help="KITTI data directory")

    parser.add_argument("-p", "--pca",
                        default=50,
                        help="Number of principle components. Default: %(default)s")

    parser.add_argument("-t", "--road_type",
                        choices=["um", "uu", "umm", "all"],
                        default="um",
                        help="Road types to consider. Default: %(default)s")

    parser.add_argument("-c", "--classification_type",
                        choices=["lane", "road"],
                        default="road",
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

    parser.add_argument('-s', '--show',
                        type=int,
                        help="Show test set image of specified number after running.")

    args = parser.parse_args()

    assert args.channels == 3

    if args.road_type != "all":
        args.road_type = args.road_type + "_"

    return args

def main():
    args = parse_args()

    load_start = time()
    print "Loading data from: ", args.data_dir
    X, Y = load_data(args)

    print "  Loading took %fs" % (time() - load_start)

    print "Splitting data into train, test, validation"
    train_validation_data, test_data, train_validation_targets, test_targets = train_test_split(X, Y, test_size=0.3, random_state=0)
    train_data, validation_data, train_targets, validation_targets = train_test_split(train_validation_data, train_validation_targets, test_size=10./70., random_state=0)


    #50 components covers 95% variation on um
    pca = decomposition.PCA(n_components=int(args.pca)) 

    print "Fitting PCA"
    pca.fit(train_data)
    print "Done"

    print "PCA Explained Variance:", np.sum(pca.explained_variance_ratio_)

    print "Transforming data"
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)
    validation_data_pca = pca.transform(validation_data)

    print "Train shape:", train_data_pca.shape
    print "Test shape:", test_data_pca.shape
    print "Validation shape:", validation_data_pca.shape

    #Save output
    print "Saving Output"
    np.save("data2/x-train-pca%s_%s_%s.npy" % (args.pca, args.classification_type, args.road_type), train_data_pca)
    np.save("data2/x-valid-pca%s_%s_%s.npy" % (args.pca, args.classification_type, args.road_type), validation_data_pca)
    np.save("data2/x-test-pca%s_%s_%s.npy" % (args.pca, args.classification_type, args.road_type), train_data_pca)
    np.save("data2/x-test_%s_%s.npy" % (args.classification_type, args.road_type), test_data)
    np.save("data2/y-train_%s_%s.npy" % (args.classification_type, args.road_type), train_targets)
    np.save("data2/y-valid_%s_%s.npy" % (args.classification_type, args.road_type), validation_targets)
    np.save("data2/y-test_%s_%s.npy" % (args.classification_type, args.road_type), test_targets)

def load_data(args):
    #Collect image paths
    train_dir = path.join(args.data_dir, "training", "image_2")
    train_images = []
    for img in os.listdir(train_dir):
        if img.startswith(args.road_type) or args.road_type == "all":
            train_images.append(path.join(train_dir,img))
    train_images.sort()

    if args.num_images == None:
        args.num_images = len(train_images)
    else:
        args.num_images = min(len(train_images), args.num_images)
    print "Processing %d input images" % (args.num_images)

    gt_dir = path.join(args.data_dir, "training", "gt_image_2")
    gt_images = []
    for img in os.listdir(gt_dir):
        if img.startswith(args.road_type) or args.road_type == "all":
            if args.classification_type in img:
                gt_images.append(path.join(gt_dir, img))
    gt_images.sort()

    #Verify names match
    assert len(train_images) == len(gt_images)
    for i in xrange(len(train_images)):
        train_img = train_images[i]
        gt_img = gt_images[i]
        gt_trimmed = gt_img.replace(args.classification_type + "_", "")
        assert path.basename(gt_trimmed) == path.basename(train_img)

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
        X[i,:] = train_pixels.reshape([npixels*args.channels])

        y[i] = gt_pixels[:,:,2].flatten()

        if args.num_images and i + 1 >= args.num_images:
            break

    return X, y

if __name__ == "__main__":
    main()
