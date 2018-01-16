"""
This script converts the bounding box information included in
the DeepFashion data set to the YOLOv2 format.
It also accounts for rescaling and cropping/padding of the images.
"""
import argparse
from functools import partial
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline

from utils.io import Labels
from preprocessing.preproc import ScaleImages, PadImages, CropImages
import pdb


def build_arg_dict(arg_list):
    arg_dict = {
        'category_file': arg_list.category_file,
        'bbox_file': arg_list.bbox_file,
        'image_dir': arg_list.image_dir,
    }
    return arg_dict


def update_bounding_box(old_shape, new_shape, row):
    """
    Takes a dataframe row containing bounding box and image
    labeled as "x_1", "y_1", "x_2", and "y_2" formatted
    for deep fashion. The function converts the bounding box
    information to match the image after conversion to the target
    shape.

    :param new_shape: The image shape after transformation
    :param row: A pandas Series as described above
    """
    # First index specifies rows of image, second index columns.
    # This means first index is y and second x
    scale_factor = new_shape[0] / old_shape[0]
    offset = (new_shape[1] - old_shape[1] * scale_factor)/2

    # May end up a little off-center
    row['y_1'] = row['y_1'] * scale_factor
    row['y_2'] = row['y_2'] * scale_factor
    row['x_1'] = row['x_1'] * scale_factor + offset
    row['x_2'] = row['x_2'] * scale_factor + offset
    return row[['x_1', 'x_2', 'y_1', 'y_2']]


def deepfash_to_yolo(image_shape, row):
    """
    Takes a dataframe row containing category and bounding box info
    labeled as "image_name", "category_label", "x_1", "y_1", "x_2", and "y_2" formatted
    for deep fashion. It then converts this to the format accepted by
    YOLO.

    :param image_shape: 2-tuple of integers
    :param row: pandas Series object
    :return: 5 element list representing YOLO style bounding box entry
    """
    # On images displayed using skimage.io.imshow, DeepFashion bounding box
    # values look like:
    #   small y
    #     ||
    #     ||
    #    \||/
    #     \/
    #   large y
    #   small x ----------> large x
    #
    # This means image_shape[0] corresponds to y values
    # The YOLO format seems to match (but use fractions)
    frac_x1 = row['x_1'] / image_shape[1]
    frac_x2 = row['x_2'] / image_shape[1]
    frac_y1 = row['y_1'] / image_shape[0]
    frac_y2 = row['y_2'] / image_shape[0]

    width_x = frac_x2 - frac_x1
    width_y = frac_y2 - frac_y1
    center_x = (frac_x1 + frac_x2)/2
    center_y = (frac_y1 + frac_y2)/2

    return [row['image_name'], row['category_label'],
            center_x, center_y, width_x, width_y]


def prep_image_data(arg_dict):
    """
    Prepares a label object with both bounding box and image file info

    :param arg_dict: Dictionary with input arguments
    """
    cat_df = pd.read_csv(arg_dict['category_file'],
                         skiprows=1,
                         sep='\s+')
    bbox_df = pd.read_csv(arg_dict['bbox_file'],
                          skiprows=1,
                          sep='\s+')
    img_dir = arg_dict['image_dir']

    combo_df = pd.merge(cat_df, bbox_df, how='outer', on='image_name')
    combo_df['image_name'] = combo_df['image_name'].apply(
                lambda x: x[len('img'):-len('.jpg')])
    labels = Labels(combo_df, img_dir, n_images_loaded=-1)
    labels.set_data_target('raw_image', chunksize=3000)
    return labels


def convert_bbox(arg_dict):
    labels = prep_image_data(arg_dict)
    target_width = 608
    standardize_img = Pipeline([
        ('scale', ScaleImages(target_width)),
        ('crop', CropImages(target_width)),
        ('pad', PadImages(target_width)),
    ])

    yolo_bbox = os.sep.join(arg_dict['bbox_file'].split(os.sep)[:-1]
                            + ['yolo_'
                               + arg_dict['bbox_file'].split(os.sep)[-1]])
    header = True
    with open(yolo_bbox, 'w') as bbox_file:
        for i in range(labels.n_chunk):
            chunk = next(labels)

            # Update the bounding box coordinates
            chunk[['x_1', 'x_2', 'y_1', 'y_2']] = chunk.apply(
                    lambda x: update_bounding_box(
                        x['data'].shape,
                        (target_width, target_width),
                        x),
                    axis=1)

            # Process the images and save them
            chunk['data'] = chunk['data'].apply(standardize_img.transform)
            labels.save(chunk)

            yolo_chunk = chunk.apply(partial(deepfash_to_yolo,
                                             (target_width, target_width)),
                                     axis=1)
            yolo_chunk = pd.DataFrame(np.stack(yolo_chunk.values),
                                      columns=[
                                          'image_name',
                                          'category_label',
                                          'center_x',
                                          'center_y',
                                          'width_x',
                                          'width_y',
                                      ])

            # Save the updated bounding box coordinates
            yolo_chunk.to_csv(bbox_file, index=False, header=header)
            if header:
                header = False
            print('Finished chunk {} of {}'.format(i+1, labels.n_chunk))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert deepfash style ground truth bounding'
                    ' boxes to YOLO style')
    parser.add_argument(
        'category_file', nargs='?',
        default='data/Category and Attribute Prediction Benchmark/Anno/list_category_img.txt',
        help='File mapping images to category number')
    parser.add_argument(
        'bbox_file', nargs='?',
        default='data/Category and Attribute Prediction Benchmark/Anno/list_bbox.txt',
        help='File mapping images to bounding box within image')
    parser.add_argument(
        'image_dir', nargs='?',
        default='data/Category and Attribute Prediction Benchmark/Img/Img/',
        help='Directory containing image references')
    args = parser.parse_args()

    arg_dict = build_arg_dict(args)
    convert_bbox(arg_dict)
