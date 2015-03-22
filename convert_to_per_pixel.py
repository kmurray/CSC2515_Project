#!/usr/bin/env python
import argparse
import os
import os.path as path
import csv

import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Convert png data to csv")

    parser.add_argument("-d", "--data_dir", 
                        required=True,
                        help="Directory containing KITTI image files")

    parser.add_argument("-r", "--radius",
                        required=True,
                        type=int,
                        help="Radius of square set of pixels to consider. Sets the size of the input feature vector.")

    parser.add_argument("-t", "--road_type",
                        choices=["um", "uu", "umm", "all"],
                        default="um",
                        help="Road types to consider. Default: %(default)s")

    parser.add_argument("-c", "--classification_type",
                        choices=["lane", "road"],
                        default="lane",
                        help="Type of classification. Default: %(default)s")

    parser.add_argument("-o", "--out_file",
                        help="Output file")

    parser.add_argument("-n", "--num_images",
                        type=int,
                        help="Override the number of images to process. Default all.")
            

    args = parser.parse_args()

    if args.road_type != "all":
        args.road_type = args.road_type + "_"

    return args

def main():

    args = parse_args()

    train_dir = path.join(args.data_dir, "training", "image_2")
    train_images = []
    for img in os.listdir(train_dir):
        if img.startswith(args.road_type) or args.road_type == "all":
            train_images.append(path.join(train_dir,img))
    train_images.sort()

    gt_dir = path.join(args.data_dir, "training", "gt_image_2")
    gt_images = []
    for img in os.listdir(gt_dir):
        if img.startswith(args.road_type) or args.road_type == "all":
            if args.classification_type in img:
                gt_images.append(path.join(gt_dir, img))
    gt_images.sort()

    print "Training:"
    print train_images
    print "Ground Truth:"
    print gt_images

    train_gt_images = []
    for i in xrange(len(train_images)):
        train_gt_images.append((train_images[i], gt_images[i]))

    convert_training_images(args, train_gt_images)

def convert_training_images(args, train_gt_images):
    colours = ['R', 'G', 'B']
    pixel_offsets = range(-args.radius+1,args.radius)

    field_names = ["Label"]
    for y_offset in pixel_offsets:
        for x_offset in pixel_offsets:
            for channel in colours:
                field_names.append(pixel_label(x_offset, y_offset, channel))

    print field_names
    with open(path.join(args.out_file), 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        csv_writer.writeheader()

        for img_cnt, (train_img_filepath, gt_img_filepath) in enumerate(train_gt_images):
            print "Converting", train_img_filepath
            img_train = Image.open(train_img_filepath)
            img_gt = Image.open(gt_img_filepath)
            w, h = img_train.size
            pixels_train = img_train.load()
            pixels_gt = img_gt.load()

            for y in xrange(h):
                for x in xrange(w):
                    pixel_data = {}
                    
                    if pixels_gt[x,y][2] == 255: #Lane label
                        pixel_data["Label"] = 1.
                    else:
                        pixel_data["Label"] = 0.

                    for y_offset in pixel_offsets:
                        for x_offset in pixel_offsets:
                            pixel = None
                            curr_x = x+x_offset
                            curr_y = y+y_offset
                            if curr_y < 0 or curr_y >= h or curr_x < 0 or curr_x >= w:
                                #Zero if out of range
                                pixel = (0, 0, 0)
                            else:
                                pixel = pixels_train[curr_x,curr_y]

                            for i, colour in enumerate(colours):
                                pixel_data[pixel_label(x_offset, y_offset, colour)] = float(pixel[i]) / 255
                    csv_writer.writerow(pixel_data)

            if args.num_images and img_cnt + 1 >= args.num_images:
                break


def pixel_label(x_offset, y_offset, channel):
    return "%dX_%dY_%s" % (x_offset, y_offset, channel)

if __name__ == "__main__":
    main()
