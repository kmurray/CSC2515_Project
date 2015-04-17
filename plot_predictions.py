#!/usr/bin/env python
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def parse_args():
    parser = argparse.ArgumentParser(description="Plot predictor results, sorted by error")

    parser.add_argument('-X',
                        help="Input Data file in .npy format",
                        required=True,
                        )
    parser.add_argument('-t',
                        help="Targets file in .npy format",
                        required=True,
                        )
    parser.add_argument('-y',
                        help="Predictions file in .npy format",
                        required=True,
                        )

    parser.add_argument('--plot_type',
                         choices=['full', 'error_matrix'],
                         default='full',
                         help="Type of plot to show. Default %(default)s. 'Full' shows the standard plot with input image, target, prediction and error. 'error_matrix' shows a matrix of errors.")
    parser.add_argument('--matrix_w',
                         type=int,
                         default=2,
                         help="Number of errors to display horizontally for 'error_matrix' plot type"
                         )
    parser.add_argument('--matrix_h',
                         type=int,
                         default=4,
                         help="Number of errors to display horizontally for 'error_matrix' plot type"
                         )
    parser.add_argument("--width",
                        type=int,
                        default=1242,
                        help="Width of images. Default: %(default)s (will be resized if different)")
    parser.add_argument("--height",
                        type=int,
                        default=375,
                        help="Height of images. Default: %(default)s (will be resized if different)")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    X = np.load(args.X)
    t = np.load(args.t)
    y = np.load(args.y)

    print "X:", X.shape
    print "t:", t.shape
    print "y:", y.shape

    err = np.empty(t.shape)
    err_frac = np.empty(t.shape[0])
    #err = t - y
    err = y - t
    for i in xrange(X.shape[0]):
        err_frac[i] = np.sum(np.abs(err[i,:])) / np.prod(err[i,:].shape)

    sorted_error_indicies = np.argsort(err_frac)

    print "Mean Error: ", np.mean(err_frac)
    print "Max Error: ", np.max(err_frac)
    print "Min Error: ", np.min(err_frac)
    for pctile in [99, 95, 90, 75]: 
        print str(pctile) + "th Percentile Error: ", np.percentile(err_frac, q=pctile)

    if args.plot_type == 'full':
        for i in reversed(sorted_error_indicies):
            draw_full(args, X, t, y, err, err_frac, i)

    elif args.plot_type == 'error_matrix':
        draw_error_matrix(args, X, t, y, err, err_frac, sorted_error_indicies)

    else:
        raise ValueError("Unsupported plot type %s" % args.plot_type)

def draw_error_matrix(args, X, t, y, err, err_frac, sorted_error_indicies):

    reversed_error_indicies = np.fliplr([sorted_error_indicies])[0]

    nimages_per_matrix = args.matrix_w*args.matrix_h
    cmap = plt.get_cmap("bwr")

    for j in xrange(0, sorted_error_indicies.shape[0], nimages_per_matrix):
        fig, axarr = plt.subplots(args.matrix_h, args.matrix_w)

        for i in xrange(nimages_per_matrix):
            x, y = np.unravel_index(i, (args.matrix_h, args.matrix_w))
            idx = reversed_error_indicies[j + i]

            err_img_flat = err[idx,:]
            err_img = err_img_flat.reshape([args.height,args.width])

            axarr[x, y].set_title("Error: %f" % err_frac[idx])
            axarr[x, y].imshow(-1*err_img, cmap=cmap)

        plt.tight_layout()
        plt.show()

def draw_full(args, X, t, y, err, err_frac, idx):

    #Reshape images into image matricies for plotting
    img_flat = X[idx,:]
    img = img_flat.reshape([args.height,args.width,3])

    gt_flat = y[idx,:]
    gt = gt_flat.reshape([args.height,args.width])

    pred_flat = t[idx,:]
    pred = pred_flat.reshape([args.height,args.width])

    err_img_flat = err[idx,:]
    err_img = err_img_flat.reshape([args.height,args.width])

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

    axarr[3].set_title("GroundTruth - Prediction\nError: %f" % err_frac[idx])
    axarr[3].imshow(err_img, cmap=cmap)

    #fig.colorbar(pred_err_show, orientation="horizontal")
    plt.tight_layout()
    plt.show()

def target_pred_error_frac(t_val, y_val):

    err_frac = np.sum(np.abs(t_val - y_val)) / np.prod(y_val.shape)

    return err_frac

if __name__ == "__main__":
    main()
