"""
Runs several k-means calculations on the width_x and width_y columns
of the supplied YOLO format bounding box data files. Outputs all anchor
box files and a plot of their MSE and silhouette score for the user to 
decide which configuration is best
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score


def build_arg_dict(args):
    arg_dict = {
            'bbox_file': args.bbox_file,
            'anchor_file_dir': args.anchor_file_dir,
            }
    return arg_dict


def write_anchors(file_name, kmeans):
    with open(file_name, 'w') as anchorf:
        anchors = [str(a) for a in np.ravel(kmeans.cluster_centers_)]
        anchorf.write(', '.join(anchors))


def kmeans_mse(vals, labels, centers):
    centers = centers[labels, :]
    mse = mean_squared_error(centers, vals)
    return mse.sum()


def plot_scores(file_name, mse_scores):
    fig, ax = plt.subplots()

    ax[0].plot(range(1, len(mse_scores)+1), mse_scores)

    ax[0].set_title('Mean Squared Error')

    ax[0].set_xlabel('Number of clusters')
    fig.savefig(file_name)
    plt.close(fig)


def fit_anchor_boxes(arg_dict):
    bbox_df = pd.read_csv(arg_dict['bbox_file'],
                          usecols=['width_x', 'width_y'])

    #bbox_df = bbox_df.head(20)
    max_means = 10
    min_means = 1
    mse_scores = []
    for i in range(min_means, max_means+1):
        kmeans = KMeans(i, precompute_distances=False)
        labels = kmeans.fit_predict(bbox_df.values)
        print('Finished kmeans for {} clusters'.format(i))

        anchor_file_name = arg_dict['anchor_file_dir']+os.sep\
                +'anchor_{}.txt'.format(i)
        write_anchors(anchor_file_name, kmeans)
        
        mse_scores.append(kmeans_mse(bbox_df.values, labels,
            kmeans.cluster_centers_))

    plot_file_name = arg_dict['anchor_file_dir']+os.sep+'anchor_plot.png'
    plot_scores(plot_file_name, mse_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run k-means against bounding box dimensions and output'
                    ' the results.')
    parser.add_argument(
        'bbox_file', nargs='?',
        default='data/Category and Attribute Prediction Benchmark/Anno/list_bbox.txt',
        help='File with YOLO style (ie. listed as a fraction of image dims)'
             ' bounding box dimensions.')
    parser.add_argument(
        'anchor_file_dir', nargs='?',
        default='anchors',
        help='Directory where anchor box files and k-means MSE plot will be'
             ' output.')
    args = parser.parse_args()

    arg_dict = build_arg_dict(args)
    fit_anchor_boxes(arg_dict)
