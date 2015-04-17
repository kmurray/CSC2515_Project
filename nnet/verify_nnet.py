#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fann2 import libfann

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

    parser.add_argument("--width",
                        type=int,
                        default=1242,
                        help="Width of images. Default: %(default)s (will be resized if different)")
    parser.add_argument("--height",
                        type=int,
                        default=375,
                        help="Height of images. Default: %(default)s (will be resized if different)")

    parser.add_argument('-s', '--show',
                        required=True,
                        type=int,
                        help="Show test set image of specified number after running.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    print "loading data"
    x_pca_test = np.load("data/x-test-pca%s_%s.npy" % (args.pca, args.type))
    x_test = np.load("data/x-test_%s.npy" % args.type)
    y_test = np.load("data/y-test_%s.npy" % args.type)

    print "x_pca_test Shape", x_pca_test.shape
    print "x_test Shape", x_test.shape
    print "y_test Shape", y_test.shape

    print "Loading fann"
    ann = libfann.neural_net()
    ann.create_from_file("nnets/%s" % args.fann_file)

    predicted = []

    for test in x_pca_test:
        predicted.append(np.array(ann.run(test)))

    print "Test Error:", get_error(predicted, y_test)

    print "Saving predictions"
    np.save("data/p-test_%s" % args.type, np.array(predicted))

    print "Smoothing predictions"
    predicted_smooth = smooth(args, predicted)

    print "Smooth Test Error:", get_error(predicted_smooth, y_test)

    print "Saving smoothed predictions"
    np.save("data/ps-test_%s" % args.type, np.array(predicted_smooth))

    plot_classification(args, x_test, y_test, np.array(predicted_smooth))

def get_error(predicted, expected):
    bit_fail = 0

    for pred, exp in zip(predicted, expected):
        for p, e in zip(pred, exp):
            if abs(p - e) > 0.5:
                bit_fail += 1

    return (float(bit_fail) / (expected.shape[0] * expected.shape[1]))

def mean(args, img, w, h):
    num_pixels = 0
    mean = 0.0
    minw = max(0, w - 2)
    maxw = min(args.width - 1, w + 2)
    minh = max(0, h - 2)
    maxh = min(args.height - 1, h + 2)

    for h in xrange(minh, maxh + 1):
        for w in xrange(minw, maxw + 1):
            num_pixels += 1
            mean += img[h][w]

    return mean / num_pixels

def smooth(args, predicted):
    out = []
    for img_flat in predicted:
        img_smooth = np.zeros((args.height, args.width))
        img = img_flat.reshape([args.height, args.width])

        for h in xrange(args.height):
            for w in xrange(args.width):
                img_smooth[h][w] = mean(args, img, w, h)

        out.append(img_smooth.reshape([args.height * args.width]))
    return out


def plot_classification(args, test_data, test_targets, predicted_targets):
    #Reshape images into image matricies for plotting
    img_flat = test_data[args.show,:]
    img = img_flat.reshape([args.height, args.width, 3])

    gt_flat = test_targets[args.show,:]
    gt = gt_flat.reshape([args.height, args.width])

    pred_flat = predicted_targets[args.show,:]
    pred = pred_flat.reshape([args.height, args.width])

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
    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
