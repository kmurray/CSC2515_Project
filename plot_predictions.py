#!/usr/bin/env python
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def parse_args():
    parser = argparse.ArgumentParser(description="Plot predictor results")

    parser.add_argument('-X',
                        help="Input Data",
                        required=True,
                        )
    parser.add_argument('-t',
                        help="Targets",
                        required=True,
                        )
    parser.add_argument('-y',
                        help="Predictions",
                        required=True,
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

    err = np.empty(t.shape)
    err_frac = np.empty(t.shape[0])
    for i in xrange(X.shape[0]):
        err[i,:] = t[i,:] - y[i,:]
        err_frac[i] = np.sum(np.abs(err[i,:])) / np.prod(err[i,:].shape)

    sorted_error_indicies = np.argsort(err_frac)

    for i in reversed(sorted_error_indicies):
        draw(args, X, t, y, err, err_frac, i)

def draw(args, X, t, y, err, err_frac, idx):

    #Reshape images into image matricies for plotting
    img_flat = X[idx,:]
    img = img_flat.reshape([args.height,args.width,3])

    gt_flat = t[idx,:]
    gt = gt_flat.reshape([args.height,args.width])

    pred_flat = y[idx,:]
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

    axarr[3].set_title("GroundTruth - Prediction\nError: %f" % err_frac[idx])
    axarr[3].imshow(gt - pred, cmap=cmap)

    #fig.colorbar(pred_err_show, orientation="horizontal")
    plt.tight_layout()
    plt.show()

def target_pred_error_frac(t_val, y_val):

    err_frac = np.sum(np.abs(t_val - y_val)) / np.prod(y_val.shape)

    return err_frac

if __name__ == "__main__":
    main()
