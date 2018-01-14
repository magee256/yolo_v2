""""
Contains the unit tests for convert_bbox.py

These tests assume the use of pytest
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from utils.preproc import ScaleImages, PadImages, CropImages

from convert_bbox import update_bounding_box, deepfash_to_yolo


def make_old_bbox(row):
    """
    Creates the lines needed to draw a bounding box from old data rep

    :param row: row from :func plot_img_and_bound_box:
    :return: bbox_lines - numpy array marking corners of bounding box
    """
    bbox_lines = np.array([
        [row['x_1'], row['y_1']],
        [row['x_1'], row['y_2']],
        [row['x_2'], row['y_2']],
        [row['x_2'], row['y_1']],
        [row['x_1'], row['y_1']],
    ])
    return bbox_lines


def make_new_bbox(yolo_dat, img_shape):
    """
    Creates the lines needed to draw a bounding box from new data rep

    :param row: row from :func plot_img_and_bound_box:
    :return: bbox_lines - numpy array marking corners of bounding box
    """
    x_1 = (yolo_dat[2] - yolo_dat[4]/2)*img_shape[1]
    y_1 = (yolo_dat[3] - yolo_dat[5]/2)*img_shape[0]
    x_2 = (yolo_dat[2] + yolo_dat[4]/2)*img_shape[1]
    y_2 = (yolo_dat[3] + yolo_dat[5]/2)*img_shape[0]
    bbox_lines = np.array([
        [x_1, y_1],
        [x_1, y_2],
        [x_2, y_2],
        [x_2, y_1],
        [x_1, y_1],
    ])
    return bbox_lines


def plot_img_and_bound_box(img, row):
    """
    Draws bounding box on image.

    For visual testing. Draws bounding box on image
    before and after processing of the image.

    :param img: The original image as a numpy array
    :param row: Row with "x_1", "y_2", etc. bbox data before transform
    :return: None, Displays plot of image before and after updating.
    """
    fig, ax = plt.subplots(1, 3)

    # Plot untransformed image on left
    ax[0].imshow(img)
    bbox = make_old_bbox(row)
    ax[0].plot(bbox[:, 0], bbox[:, 1])

    # Transform image and data
    target_width = 248
    update_bounding_box(row['data'].shape, (target_width, target_width), row)

    standardize_img = Pipeline([
        ('scale', ScaleImages(target_width)),
        ('crop', CropImages(target_width)),
        ('pad', PadImages(target_width)),
    ])
    img = standardize_img.transform(img)

    # Plot transformed image in middle
    ax[1].imshow(img)
    bbox = make_old_bbox(row)
    ax[1].plot(bbox[:, 0], bbox[:, 1])

    # Convert data to YOLO format
    yolo_dat = deepfash_to_yolo(img.shape, row)

    # Plot image using YOLO style coordinates
    ax[2].imshow(img)
    bbox = make_new_bbox(yolo_dat, img.shape)
    ax[2].plot(bbox[:, 0], bbox[:, 1])
    plt.show()


def test_update_bounding_box():
    # First index specifies rows of image, second index columns.
    # This means first index is y and second x
    base_shape = (300, 400, 3)
    no_change = pd.Series({
        "data": np.ones(base_shape),
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    update_bounding_box(base_shape, base_shape, no_change)
    assert no_change["x_1"] == 50
    assert no_change["y_1"] == 70
    assert no_change["x_2"] == 250
    assert no_change["y_2"] == 300

    scale_only = pd.Series({
        "data": np.ones(base_shape),
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    update_bounding_box(base_shape, (600, 800, 3), scale_only)
    assert scale_only["x_1"] == 100
    assert scale_only["y_1"] == 140
    assert scale_only["x_2"] == 500
    assert scale_only["y_2"] == 600

    crop_only = pd.Series({
        "data": np.ones(base_shape),
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    update_bounding_box(base_shape, (300, 300, 3), crop_only)
    assert crop_only["x_1"] == 0
    assert crop_only["y_1"] == 70
    assert crop_only["x_2"] == 200
    assert crop_only["y_2"] == 300

    pad_only = pd.Series({
        "data": np.ones(base_shape),
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    update_bounding_box(base_shape, (300, 500, 3), pad_only)
    assert pad_only["x_1"] == 100
    assert pad_only["y_1"] == 70
    assert pad_only["x_2"] == 300
    assert pad_only["y_2"] == 300

    scale_crop = pd.Series({
        "data": np.ones(base_shape),
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    update_bounding_box(base_shape, (600, 600, 3), scale_crop)
    assert scale_crop["x_1"] == 0
    assert scale_crop["y_1"] == 140
    assert scale_crop["x_2"] == 400
    assert scale_crop["y_2"] == 600

    scale_pad = pd.Series({
        "data": np.ones(base_shape),
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    update_bounding_box(base_shape, (600, 1000, 3), scale_pad)
    assert scale_pad["x_1"] == 200
    assert scale_pad["y_1"] == 140
    assert scale_pad["x_2"] == 600
    assert scale_pad["y_2"] == 600


def test_deepfash_to_yolo():
    test_row = pd.Series({
        "category_label": 1,
        "image_name": "fake_name.jpg",
        "x_1": 50,
        "y_1": 70,
        "x_2": 250,
        "y_2": 300,
    })
    convert = deepfash_to_yolo((300, 400, 3), test_row)
    tol = 1e-8
    assert convert[0] == "fake_name.jpg"
    assert convert[1] == 1
    assert np.isclose(convert[2], 0.375, atol=tol)
    assert np.isclose(convert[3], (70 + (300 - 70)/2)/300, atol=tol)
    assert np.isclose(convert[4], (250 - 50)/400, atol=tol)
    assert np.isclose(convert[5], (300 - 70)/300, atol=tol)


def test_scale_images():
    # Test with 10 images
    tst_images = np.ones((10, 540, 301, 3))
    tst_df = pd.DataFrame([[arr] for arr in tst_images])
    half_scale = ScaleImages(270)
    double_scale = ScaleImages(1080)

    # Test on a single image
    doubled = double_scale.transform(tst_images[1, :, :, :])
    halved = half_scale.transform(tst_images[1, :, :, :])
    assert doubled.shape[1] == 602
    assert halved.shape[1] == 150
    assert doubled.shape[0] == 1080
    assert halved.shape[0] == 270

    # Test applying it to a DataFrame
    double_df = tst_df.iloc[:, 0].apply(double_scale.transform)
    half_df = tst_df.iloc[:, 0].apply(half_scale.transform)
    assert double_df.iloc[4].shape[1] == 602
    assert half_df.iloc[4].shape[1] == 150
    assert double_df.iloc[4].shape[0] == 1080
    assert half_df.iloc[4].shape[0] == 270


def test_pad_images():
    # Test with 10 images
    correct_dim = np.ones((10, 540, 540, 3))
    need_pad = np.ones((10, 540, 370, 3))
    no_pad = np.ones((10, 540, 600, 3))

    correct_dim_df = pd.DataFrame([[arr] for arr in correct_dim])
    need_pad_df = pd.DataFrame([[arr] for arr in need_pad])
    no_pad_df = pd.DataFrame([[arr] for arr in no_pad])

    pad_image = PadImages(540)

    # Test on a single image
    correct = pad_image.transform(correct_dim[1, :, :, :])
    with_pad = pad_image.transform(need_pad[1, :, :, :])
    wo_pad = pad_image.transform(no_pad[1, :, :, :])
    assert correct.shape[0] == 540
    assert correct.shape[1] == 540
    assert with_pad.shape[0] == 540
    assert with_pad.shape[1] == 540
    assert wo_pad.shape[0] == 540
    assert wo_pad.shape[1] == 600

    # Test applying it to a DataFrame
    correct = correct_dim_df.iloc[:, 0].apply(pad_image.transform)
    with_pad = need_pad_df.iloc[:, 0].apply(pad_image.transform)
    wo_pad = no_pad_df.iloc[:, 0].apply(pad_image.transform)
    assert correct.iloc[4].shape[0] == 540
    assert correct.iloc[4].shape[1] == 540
    assert with_pad.iloc[4].shape[0] == 540
    assert with_pad.iloc[4].shape[1] == 540
    assert wo_pad.iloc[4].shape[0] == 540
    assert wo_pad.iloc[4].shape[1] == 600


def test_crop_images():
    # Test with 10 images
    correct_dim = np.ones((10, 540, 540, 3))
    no_crop = np.ones((10, 540, 370, 3))
    need_crop = np.ones((10, 540, 600, 3))

    correct_dim_df = pd.DataFrame([[arr] for arr in correct_dim])
    need_pad_df = pd.DataFrame([[arr] for arr in no_crop])
    no_pad_df = pd.DataFrame([[arr] for arr in need_crop])

    pad_image = CropImages(540)

    # Test on a single image
    correct = pad_image.transform(correct_dim[1, :, :, :])
    wo_crop = pad_image.transform(no_crop[1, :, :, :])
    with_crop = pad_image.transform(need_crop[1, :, :, :])
    assert correct.shape[0] == 540
    assert correct.shape[1] == 540
    assert wo_crop.shape[0] == 540
    assert wo_crop.shape[1] == 370
    assert with_crop.shape[0] == 540
    assert with_crop.shape[1] == 540

    # Test applying it to a DataFrame
    correct = correct_dim_df.iloc[:, 0].apply(pad_image.transform)
    wo_crop = need_pad_df.iloc[:, 0].apply(pad_image.transform)
    with_crop = no_pad_df.iloc[:, 0].apply(pad_image.transform)
    assert correct.iloc[4].shape[0] == 540
    assert correct.iloc[4].shape[1] == 540
    assert wo_crop.iloc[4].shape[0] == 540
    assert wo_crop.iloc[4].shape[1] == 370
    assert with_crop.iloc[4].shape[0] == 540
    assert with_crop.iloc[4].shape[1] == 540
