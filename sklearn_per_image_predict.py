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

from sklearn import preprocessing, decomposition, lda
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

    parser.add_argument('--show_test_error',
                        default=True,
                        action='store_true',
                        help="Show the test set error")

    parser.add_argument('-s', '--show',
                        type=int,
                        help="Show test set image of specified number after running.")

    parser.add_argument('--plot_eigenroads',
                        type=int,
                        metavar="N",
                        help="Plot the top N eigenroads")

    parser.add_argument('--classifier_type',
                        choices=['kNN', 'DecisionTree', 'EnsembleForest', 'ExtraTrees'], #'AdaBoost', 'GradBoost', 
                        help="Classifier type.")
    parser.add_argument('--classifier_params',
                        default=None,
                        help="Parameters for the specified classifier. Format: 'key1=value1 key2=value2'")

    parser.add_argument('--dimreducer_type',
                        choices=['PCA',],
                        default='PCA',
                        help="What form of dimension reduction to use. Must be one of %(choices)s. Default: %(default)s")
    parser.add_argument('--dimreducer_params',
                        default='n_components=200',
                        help="Number of components to preserve for PCA or LDA. Default: \"%(default)s\"")

    args = parser.parse_args()

    assert args.channels == 3

    if args.road_type != "all":
        args.road_type = args.road_type + "_"

    if args.classifier_params != None:
        print "Classifier Params: " + args.classifier_params
        args.classifier_params_dict = str2dict(args.classifier_params)
    else:
        args.classifier_params_dict = {}

    args.dimreducer_params_dict = str2dict(args.dimreducer_params)

    return args

def str2dict(input_str):
    out_dict = {}

    key_values = input_str.split()

    for key_value in key_values:
        key_str, value_str = key_value.split('=')

        #Try to convert value as appropriate
        value = value_str
        try: value = int(value_str)
        except ValueError:
            try: value = float(value_str)
            except ValueError:
                pass #Default to string

        out_dict[key_str] = value
    return out_dict

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

    train_data, validation_data, train_targets, validation_targets = train_test_split(train_validation_data, train_validation_targets, test_size=10./70., random_state=0)
    #train_data, train_targets = train_validation_data, train_validation_targets
    print "  Train %d, Valid %d, Test %d" % (train_data.shape[0], validation_data.shape[0], test_data.shape[0])

    estimator = None
    if args.classifier_type == 'kNN':
        #kNN params: 
        # n_neighbors=int, number of neighbours to weight over
        # weights=['uniform', 'distance'], how to weight neighbours: uniform is majority vote
        # p=int, (What norm are we using, defaults to l2
        classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    elif args.classifier_type == 'DecisionTree':
        classifier = tree.DecisionTreeClassifier(max_depth=3)
    elif args.classifier_type == 'EnsembleForest':
        classifier = ensemble.RandomForestClassifier(n_estimators=10)
    #elif args.classifier_type == 'AdaBoost':
        ##Broken for multi-output
        #base_classifier = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=20) #Seem reasonable based on UMM
        #classifier = ensemble.AdaBoostClassifier(base_estimator=base_classifier)
    #elif args.classifier_type == 'GradBoost':
        ##Broken for multi-output
        #classifier = ensemble.GradientBoostingClassifier()
    elif args.classifier_type == 'ExtraTrees':
        classifier = ensemble.ExtraTreesClassifier()
    #elif args.classifier_type == 'Bagging':
        #classifier = ensemble.BaggingClassifier()
    else:
        raise ValueError("Unrecognized classifier type: %s" % args.classifier_type)

    dimreducer = None
    if args.dimreducer_type == "PCA":
        dimreducer = decomposition.PCA()
    else:
        raise ValueError("Unrecognized dimension reduction type: %s" % args.dimreducer_type)

    #Override default params from commandline
    classifier.set_params(**args.classifier_params_dict)
    dimreducer.set_params(**args.dimreducer_params_dict)

    #Train
    estimator = train_estimator(args, 
                                classifier, 
                                dimreducer,
                                train_data, 
                                train_targets, 
                                validation_data, 
                                validation_targets)

    #Predict
    train_targets_pred, validation_targets_pred, test_targets_pred = pred_estimator(args, 
                                                                                    estimator, 
                                                                                    train_data, 
                                                                                    validation_data, 
                                                                                    test_data)

    #Eval
    train_err_frac, validation_err_frac, test_err_frac = eval_pred(args, 
                                                                    train_targets_pred, 
                                                                    validation_targets_pred, 
                                                                    test_targets_pred,
                                                                    train_targets,
                                                                    validation_targets,
                                                                    test_targets)
    print_error(train_err_frac, "Train")

    print_error(validation_err_frac, "Validation")

    np.save("train_data.npy", train_data)
    np.save("validation_data.npy", validation_data)
    np.save("test_data.npy", test_data)

    np.save("train_targets.npy", train_targets)
    np.save("validation_targets.npy", validation_targets)
    np.save("test_targets.npy", test_targets)

    np.save("train_pred.npy", train_targets_pred)
    np.save("validation_pred.npy", validation_targets_pred)
    np.save("test_pred.npy", test_targets_pred)

    if args.show_test_error:
        print_error(test_err_frac, "Test")
        plot_classification(args, classifier, test_data, test_targets, test_targets_pred)
    else:
        plot_classification(args, classifier, validation_data, validation_targets, validation_targets_pred)

def print_error(err_frac, err_type):
    print err_type + " Mean: ", np.mean(err_frac)
    print err_type + " Min : ", np.min(err_frac)
    print err_type + " Max : ", np.max(err_frac)
    for pctile in [99, 95, 90, 75]: 
        print err_type + " " + str(pctile) + "th Percentile Error: ", np.percentile(err_frac, q=pctile)

def train_estimator(args, classifier, dimreducer, train_data, train_targets, validation_data, validation_targets):
    print "Training Classifier"
    train_start = time()

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
            ('dimreduction', dimreducer), #First PCA
            ('classifier', classifier) #Then classify on the reduced dimension problem
        ])

    scorer = metrics.make_scorer(abs_pixel_diff_scorer, greater_is_better=False)

    #estimator = GridSearchCV(pipe,  dict(dimreduction__n_components=n_components))
    #estimator = GridSearchCV(pipe, dict(classifier__n_heighbors=[1, 2, 3]), scoring=scorer, cv=2)
    #estimator = RandomizedSearchCV(pipe,  {'classifier__n_heighbors': [1, 2]})
    estimator = pipe
    estimator.fit(train_data, train_targets)

    if args.plot_eigenroads:
        plot_eigenroads(args, estimator.steps[0][1]) #Pass fitted PCA object

    

    if args.dimreducer_type == "PCA":
        print "  PCA Explained Variance: ", np.sum(dimreducer.explained_variance_ratio_)
    else:
        print "  Picked %d components" % (dimreducer.transform(validation_data).shape[1])


    print "  Training took %fs" % (time() - train_start)

    return estimator

def pred_estimator(args, estimator, train_data, validation_data, test_data):
    print "Predicting Targets"
    pred_start = time()

    train_targets_pred = estimator.predict(train_data)
    validation_targets_pred = estimator.predict(validation_data)
    test_targets_pred = estimator.predict(test_data)

    print "  Prediction took %fs" % (time() - pred_start)

    return train_targets_pred, validation_targets_pred, test_targets_pred

def eval_pred(args, train_targets_pred, validation_targets_pred, test_targets_pred, train_targets, validation_targets, test_targets):
    print "Evaluating Predictions"
    eval_start = time()

    train_err = train_targets_pred - train_targets 
    validation_err = validation_targets_pred - validation_targets 
    test_err = test_targets_pred - test_targets 

    train_err_frac = np.sum(np.abs(train_err), 1) / train_err.shape[1]
    test_err_frac = np.sum(np.abs(test_err), 1) / train_err.shape[1]
    validation_err_frac = np.sum(np.abs(validation_err), 1) / train_err.shape[1]

    print "  Evaluation took %fs" % (time() - eval_start)

    return train_err_frac, validation_err_frac, test_err_frac

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

def plot_eigenroads(args, pca):
    neig_roads = 3

    fig, axarr = plt.subplots(neig_roads, 1)
    
    for i in xrange(neig_roads):
        eigroad_img = pca.components_[i].reshape([args.height,args.width,3])

        #re-normalize
        eigroad_img_norm = None
        if eigroad_img.min() < 0:
            eigroad_img_norm = eigroad_img - eigroad_img.min()

        eigroad_img_norm = eigroad_img_norm / eigroad_img_norm.max()

        axarr[i].set_title("EigenRoad %d. Described Variance %.2f%%" %(i, pca.explained_variance_ratio_[i]*100))
        axarr[i].imshow(eigroad_img_norm)

    plt.tight_layout()
    plt.show()



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
