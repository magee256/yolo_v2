import numpy as np
import pytest

from train_yolo import expand_truth_vals


def test_expand_truth_vals():
    n_classes = 5
    grid_dims = (13, 13)
    n_anchors = 5

    ground_truth = np.array([[1, .5, .5, .5, .5]])
    truth_array = expand_truth_vals(ground_truth, n_classes, grid_dims, n_anchors)
    assert (truth_array[0, 6, 6, :, :] == truth_array[0, 6, 6, 0, :]).all()
    assert (truth_array[0, 6, 6, 0, 0:2] == .5).all()
    assert np.isclose(truth_array[0, 6, 6, 0, 2:4], np.sqrt(.5)).all()
    assert (truth_array[0, 6, 6, 0, 4] == 1).all()
    assert (truth_array[0, 6, 6, 0, 5] == 1).all()

    ground_truth = np.array([[1, .5, .5, .5, .5],
                             [5, .25, .25, .2, .1],
                             [1, .5, .5, .5, .5, 2, .75, .75, .75, .75]])
    truth_array = expand_truth_vals(ground_truth, n_classes, grid_dims, n_anchors)
    assert (truth_array[0, 6, 6, :, :] == truth_array[0, 6, 6, 0, :]).all()
    assert (truth_array[0, 6, 6, 0, 0:2] == .5).all()
    assert np.isclose(truth_array[0, 6, 6, 0, 2:4], np.sqrt(.5)).all()
    assert (truth_array[0, 6, 6, 0, 4] == 1)
    assert (truth_array[0, 6, 6, 0, 5] == 1)

    assert (truth_array[1, 3, 3, :, :] == truth_array[1, 3, 3, 0, :]).all()
    assert (truth_array[1, 3, 3, 0, 0:2] == .25).all()
    assert np.isclose(truth_array[1, 3, 3, 0, 2], np.sqrt(.2))
    assert np.isclose(truth_array[1, 3, 3, 0, 3], np.sqrt(.1))
    assert (truth_array[1, 3, 3, 0, 4] == 1)
    assert (truth_array[1, 3, 3, 0, 9] == 1)

    assert (truth_array[2, 6, 6, :, :] == truth_array[2, 6, 6, 0, :]).all()
    assert (truth_array[2, 6, 6, 0, 0:2] == .5).all()
    assert np.isclose(truth_array[2, 6, 6, 0, 2:4], np.sqrt(.5)).all()
    assert (truth_array[2, 6, 6, 0, 4] == 1)
    assert (truth_array[2, 6, 6, 0, 5] == 1)
    assert (truth_array[2, 9, 9, :, :] == truth_array[2, 9, 9, 0, :]).all()
    assert (truth_array[2, 9, 9, 0, 0:2] == .75).all()
    assert np.isclose(truth_array[2, 9, 9, 0, 2:4], np.sqrt(.75)).all()
    assert (truth_array[2, 9, 9, 0, 4] == 1)
    assert (truth_array[2, 9, 9, 0, 6] == 1)
