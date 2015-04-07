#!/usr/bin/env python
import argparse
import os
import os.path as path
import csv
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from time import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import Image

from sklearn import preprocessing, decomposition
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, ParameterSampler
from sklearn import svm, neighbors, tree, ensemble
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
    X, y = load_data(args)

    #Save output
    np.save("X.npy", X)
    np.save("y.npy", y)
    print "  Loading took %fs" % (time() - load_start)

    print "Splitting data into train, test, validation"
    train_validation_data, test_data, train_validation_targets, test_targets = train_test_split(X, y, test_size=0.3, random_state=0)

    #train_data, validation_data, train_targets, validation_targets = train_test_split(train_validation_data, train_validation_targets, test_size=10./70., random_state=0)
    train_data, train_targets = train_validation_data, train_validation_targets
    print "  Train %d, Test %d" % (train_data.shape[0], test_data.shape[0])

    print "Training Classifier"
    train_start = time()
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    #classifier = tree.DecisionTreeClassifier(max_depth=3)
    #classifier = ensemble.RandomForestClassifier(n_estimators=10)

    #50 components covers 95% variation on um
    #50 components covers 76% variation on all
    #100 components cover 88% variation on all
    #500 components cover 100% variation on all
    #250 components cover 100% variation on all
    #150 components cover 95% variation on all
    #200 components cover 99.9% variation on all
    pca = decomposition.PCA(n_components=200) 

    n_components = [20, 40, 1000]

    pipe = Pipeline(steps=[
            ('dimreduction', pca), #First PCA
            ('classifier', classifier) #Then classify on the reduced dimension problem
        ])

    scorer = metrics.make_scorer(abs_pixel_diff_scorer, greater_is_better=False)

    #estimator = GridSearchCV(pipe,  dict(dimreduction__n_components=n_components))
    #estimator = GridSearchCV(pipe, dict(classifier__n_heighbors=[1, 2, 3]), scoring=scorer, cv=2)
    #estimator = RandomizedSearchCV(pipe,  {'classifier__n_heighbors': [1, 2]})
    estimator = pipe
    estimator.fit(train_data, train_targets)

    print "  PCA Explained Variance: ", np.sum(pca.explained_variance_ratio_)
    #print "  Picked %d components" % (estimator.best_estimator_.named_steps['dimreduction'].n_components)


    print "  Training took %fs" % (time() - train_start)

    #Evaluation
    print "Evaluating Classifier"
    eval_start = time()
    test_targets_pred = estimator.predict(test_data)
    print "  Evaluation took %fs" % (time() - eval_start)

    #Difference of test and actual
    err_frac = np.sum(np.abs(test_targets - test_targets_pred)) / np.prod(test_targets.shape)
    print "Error Rate (pixel diff): %f" % err_frac

    #print "Error Rate (cross-val): %f" % cross_val_score(estimator, X, y)

    plot_classification(args, classifier, test_data, test_targets, test_targets_pred)
    #plot_classification_interactive(args, classifier, test_data, test_targets, test_targets_pred)

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

def plot_classification(args, classifier, test_data, test_targets, predicted_targets):
    if args.show != None:

        #Reshape images into image matricies for plotting
        img_flat = test_data[args.show,:]
        img = img_flat.reshape([args.height,args.width,3])

        gt_flat = test_targets[args.show,:]
        gt = gt_flat.reshape([args.height,args.width])

        pred_flat = predicted_targets[args.show,:]
        pred = pred_flat.reshape([args.height,args.width])

        #Grid of subplots
        fig, axarr = plt.subplots(4, 1)

        #Use a divergent colour map with white as zero
        cmap = plt.get_cmap("bwr")

        axarr[0].set_title("Test Image")
        axarr[0].imshow(img)

        axarr[1].set_title("GroundTruth")
        axarr[1].imshow(gt, cmap=cmap)

        axarr[2].set_title("Prediction")
        axarr[2].imshow(pred, cmap=cmap)

        axarr[3].set_title("GroundTruth - Prediction Error")
        axarr[3].imshow(gt - pred, cmap=cmap)

        #fig.colorbar(pred_err_show, orientation="horizontal")
        plt.tight_layout()
        plt.show()

def plot_classification_interactive(args, classifier, test_data, test_targets, predicted_targets):
    callback = DrawErrors(args, test_data, test_targets, predicted_targets)

    #Grid of subplots
    fig, axarr = plt.subplots(4, 1)

    #Use a divergent colour map with white as zero
    cmap = plt.get_cmap("bwr")

    bnext = Button(axarr[0], 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axarr[0], 'Previous')
    bprev.on_clicked(callback.prev)

    plt.tight_layout()
    plt.show()

class DrawErrors():
    def __init__(self, args, test_data, test_targets, predicted_targets):
        self.idx = 0
        self.args = args
        self.test_data = test_data
        self.test_targets = test_targets
        self.predicted_targs = predicted_targets

    def draw(self):
        #Reshape images into image matricies for plotting
        img_flat = self.test_data[self.idx,:]
        img = img_flat.reshape([args.height,args.width,3])

        gt_flat = self.test_targets[self.idx,:]
        gt = gt_flat.reshape([args.height,args.width])

        pred_flat = self.predicted_targets[self.idx,:]
        pred = pred_flat.reshape([args.height,args.width])

        #Use a divergent colour map with white as zero
        cmap = plt.get_cmap("bwr")

        self.axarr[0].set_title("Test Image")
        self.axarr[0].imshow(img)

        self.axarr[1].set_title("GroundTruth")
        self.axarr[1].imshow(gt, cmap=cmap)

        self.axarr[2].set_title("Prediction")
        self.axarr[2].imshow(pred, cmap=cmap)

        self.axarr[3].set_title("GroundTruth - Prediction Error")
        self.axarr[3].imshow(gt - pred, cmap=cmap)

        #fig.colorbar(pred_err_show, orientation="horizontal")
        plt.tight_layout()
        plt.draw()

    def next(self, event):
        self.idx += 1
        self.draw()
        

    def prev(self, event):
        self.idx -= 1
        self.draw()

def abs_pixel_diff_scorer(estimator, X, y):
    y_pred = estimator.pred(X)

    err_frac = np.sum(np.abs(y - y_pred)) / np.prod(y.shape)

if __name__ == "__main__":
    main()
