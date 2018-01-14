import numpy as np
import pytest
import tensorflow as tf

from training.yolo_loss import YoloLoss
from training.train_yolo import expand_truth_vals


def evaluate_results(func):
    """
    Decorator that evaluates all Tensorflow objects returned by a function.

    Requires a default session to be set.
    """
    def convert_return(*args, **kwargs):
        tf_ret = func(*args, **kwargs)

        # Handle multi and single variable returns
        try:
            pairs = enumerate(tf_ret)
        except TypeError:
            if hasattr(tf_ret, 'eval'):
                tf_ret = tf_ret.eval()
        else:
            tf_ret = list(tf_ret)
            for i, _ in pairs:
                if hasattr(tf_ret[i], 'eval'):
                    tf_ret[i] = tf_ret[i].eval()
        return tf_ret
    return convert_return


@pytest.fixture(scope='module', autouse=True)
def set_default_tf_session():
    sess = tf.Session()
    with sess.as_default():
        yield


@pytest.fixture(autouse=True)
def record_random_seed(capsys):
    seed = np.random.randint(10000)
    with capsys.disabled():
        print('TEST RANDOM SEED: ', seed)
    np.random.seed(seed)


@pytest.fixture(scope='module', autouse=True)
def loss():
    anchors = [(4, 4),
               (2.5, 4),
               (6.5, 4),
               (.5, .5),
               (11, 11)]
    grid_dims = [13, 13]
    n_classes = 5
    pytest.loss = YoloLoss(anchors, n_classes, grid_dims)


def test_tf_test_interface():
    """
    Makes sure the convenience functions used for testing work properly.
    Sometimes tests need tests too.
    """
    A = tf.constant(4)
    B = tf.constant(6)
    C = tf.multiply(A, B)

    @evaluate_results
    def decorator_test_single(t):
        return t

    @evaluate_results
    def decorator_test_multi(t, a):
        return t, a

    c = decorator_test_single(C)
    assert c == 24

    c, a = decorator_test_multi(C, A)
    assert c == 24
    assert a == 4

    c, a = decorator_test_multi(C, 4)
    assert c == 24
    assert a == 4


def make_mock_predicted_vals(target_shape):
    """
    Creates an array with random entries suitable for testing

    :param target_shape:
    :return: pred_vals
    """
    assert len(target_shape) == 5, 'Output should be'
    " samples x width x height x anchors x preds"
    assert target_shape[4] > 5, "Must have at least 1 class to predict"

    pred_vals = np.random.random(target_shape)

    # Handle xy
    pred_vals[:, :, :, :, 0:2] = 10*pred_vals[:, :, :, :, 0:2] - 5
    pred_vals[:, :, :, :, 0:2] = pred_vals[:, :, :, :1, 0:2]

    # Handle wh
    pred_vals[:, :, :, :, 2:4] = 10*pred_vals[:, :, :, :, 2:4] - 5
    pred_vals[:, :, :, :, 2:4] = pred_vals[:, :, :, :1, 2:4]
    return pred_vals.astype(np.float32)


def prepare_init_values(ground_truth):
    truth_array = expand_truth_vals(ground_truth,
                                    pytest.loss.n_classes,
                                    pytest.loss.shape[1:3],
                                    pytest.loss.n_anchors)
    pred_vals = make_mock_predicted_vals(truth_array.shape)
    return truth_array, pred_vals


def test_convert_model_outputs(loss):
    """
    This test uses random data, may lead to intermittent errors.
    """
    pred_vals = make_mock_predicted_vals([1000, *pytest.loss.shape[1:]])

    evaled = evaluate_results(pytest.loss._convert_model_outputs)
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob = evaled(pred_vals)

    # Tests each xy falls within correct grid cell
    # xy swapped because interpreting image as row x column grid means
    # as rows change y changes.
    c_y = np.arange(pytest.loss.shape[1])
    c_x = np.arange(pytest.loss.shape[2])
    c_xy_test = np.empty((1, len(c_y), len(c_x), 1, 2))

    c_x = c_x.reshape((1, 1, len(c_x), 1, 1))
    c_y = c_y.reshape((1, len(c_y), 1, 1, 1))
    c_xy_test[:, :, :, :, 0:1] = c_x
    c_xy_test[:, :, :, :, 1:2] = c_y

    c_xy = evaluate_results(pytest.loss._create_offset_map)()
    assert ((c_xy_test - c_xy) == 0).all()
    assert ((np.floor(pred_box_xy[:, :, :, :, 0:1]*pytest.loss.shape[2]) - c_x) == 0).all()
    assert ((np.floor(pred_box_xy[:, :, :, :, 1:2]*pytest.loss.shape[1]) - c_y) == 0).all()

    # Tests wh are ordered correctly after conversion and non-negative
    # Note the predicted wh values can be larger than the image
    assert (pred_box_wh[:, :, :, :, :] >= 0).all()
    assert (pred_box_wh[:, :, :, 0, 0] >  pred_box_wh[:, :, :, 1, 0]).all()
    assert (pred_box_wh[:, :, :, 0, 1] == pred_box_wh[:, :, :, 1, 1]).all()
    assert (pred_box_wh[:, :, :, 0, 0] <  pred_box_wh[:, :, :, 2, 0]).all()
    assert (pred_box_wh[:, :, :, 0, 1] == pred_box_wh[:, :, :, 2, 1]).all()
    assert (pred_box_wh[:, :, :, 0, :] >  pred_box_wh[:, :, :, 3, :]).all()
    assert (pred_box_wh[:, :, :, 0, :] <  pred_box_wh[:, :, :, 4, :]).all()

    # Test object confidence
    assert ((pred_box_conf <= 1) & (pred_box_conf >= 0)).all()

    # Tests probabilities valid applied correctly
    assert (np.abs(pred_box_prob.sum(axis=-1) - 1.0) < 1e-6).all()


@evaluate_results
def get_areas(truth_array, pred_vals):
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob = \
        pytest.loss._convert_model_outputs(pred_vals)
    true_area, pred_area, inter_area = pytest.loss._calc_all_areas(
        truth_array[:, :, :, :, 0:2],
        truth_array[:, :, :, :, 2:4],
        pred_box_xy,
        pred_box_wh)
    return true_area, pred_area, inter_area


def test_area_calculation(loss):
    tol = 1e-4
    ground_truth = np.array([[1, .5, .5, .5, .5],
                            [5, .25, .25, .2, .1]])
    truth_array, pred_vals = prepare_init_values(ground_truth)
    grid_w, grid_h = pytest.loss.shape[1:3]

    def width_inv(target_w, anchor_ind):
        pred_w = np.log((target_w*grid_w)/pytest.loss.anchors[anchor_ind][0])
        return pred_w

    def height_inv(target_h, anchor_ind):
        pred_h = np.log((target_h*grid_h)/pytest.loss.anchors[anchor_ind][1])
        return pred_h


    # GROUND TRUTH 1
    # Sets wh to 0, avoids random numbers leading to the best bbox
    pred_vals[0, 6, 6, :, 2:4] = -200

    # Box matches exactly
    pred_vals[0, 6, 6, 0, 0] = 0  # x_vals - denotes box center
    pred_vals[0, 6, 6, 0, 1] = 0  # y_vals - denotes box center
    pred_vals[0, 6, 6, 0, 2] = width_inv(0.5, 0)
    pred_vals[0, 6, 6, 0, 3] = height_inv(0.5, 0)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[0, 6, 6, 0] - (.5*13)**2) < tol
    assert np.abs(pred_area[0, 6, 6, 0] - (.5*13)**2) < tol
    assert np.abs(inter_area[0, 6, 6, 0] - (.5*13)**2) < tol


    # Box too small
    pred_vals[0, 6, 6, 0, 0] = 0
    pred_vals[0, 6, 6, 0, 1] = 0
    pred_vals[0, 6, 6, 0, 2] = width_inv(0.25, 0)
    pred_vals[0, 6, 6, 0, 3] = height_inv(0.25, 0)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[0, 6, 6, 0] - (.5*13)**2) < tol
    assert np.abs(pred_area[0, 6, 6, 0] - (.25*13)**2) < tol
    assert np.abs(inter_area[0, 6, 6, 0] - (.25*13)**2) < tol


    # Box too large and rectangular
    pred_vals[0, 6, 6, 0, 0] = 0
    pred_vals[0, 6, 6, 0, 1] = 0
    pred_vals[0, 6, 6, 0, 2] = width_inv(0.75, 0)
    pred_vals[0, 6, 6, 0, 3] = height_inv(0.5, 0)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[0, 6, 6, 0] - (.5*13)**2) < tol
    assert np.abs(pred_area[0, 6, 6, 0] - .75*.5*13**2) < tol
    assert np.abs(inter_area[0, 6, 6, 0] - (.5*13)**2) < tol


    # Box right size but off-center
    pred_vals[0, 6, 6, 0, 0] = -np.log(3)
    pred_vals[0, 6, 6, 0, 1] = -np.log(3)
    pred_vals[0, 6, 6, 0, 2] = width_inv(0.5, 0)
    pred_vals[0, 6, 6, 0, 3] = height_inv(0.5, 0)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[0, 6, 6, 0] - (.5*13)**2) < tol
    assert np.abs(pred_area[0, 6, 6, 0] - (.5*13)**2) < tol
    assert np.abs(inter_area[0, 6, 6, 0] - 6.25**2) < tol


    # Exact match with rectangular anchor box
    pred_vals[0, 6, 6, 1, 0] = 0
    pred_vals[0, 6, 6, 1, 1] = 0
    pred_vals[0, 6, 6, 1, 2] = width_inv(0.5, 1)
    pred_vals[0, 6, 6, 1, 3] = height_inv(0.5, 1)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[0, 6, 6, 1] - (.5*13)**2) < tol
    assert np.abs(pred_area[0, 6, 6, 1] - (.5*13)**2) < tol
    assert np.abs(inter_area[0, 6, 6, 1] - (.5*13)**2) < tol


    # GROUND TRUTH 2
    # Sets wh to 0, avoids random numbers leading to the best bbox
    pred_vals[1, 3, 3, :, 2:4] = -200

    # Box matches exactly
    pred_vals[1, 3, 3, 2, 0] = -np.log(3)  # evaluates to .25
    pred_vals[1, 3, 3, 2, 1] = -np.log(3)  # evaluates to .25
    pred_vals[1, 3, 3, 2, 2] = width_inv(0.2, 2)
    pred_vals[1, 3, 3, 2, 3] = height_inv(0.1, 2)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[1, 3, 3, 2] - .1*.2*13**2) < tol
    assert np.abs(pred_area[1, 3, 3, 2] - .1*.2*13**2) < tol
    assert np.abs(inter_area[1, 3, 3, 2] - .1*.2*13**2) < tol


    # Box too small
    pred_vals[1, 3, 3, 2, 0] = -np.log(3)  # evaluates to .25
    pred_vals[1, 3, 3, 2, 1] = -np.log(3)  # evaluates to .25
    pred_vals[1, 3, 3, 2, 2] = width_inv(0.1, 2)
    pred_vals[1, 3, 3, 2, 3] = height_inv(0.1, 2)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[1, 3, 3, 2] - .1*.2*13**2) < tol
    assert np.abs(pred_area[1, 3, 3, 2] - .1*.1*13**2) < tol
    assert np.abs(inter_area[1, 3, 3, 2] - .1*.1*13**2) < tol


    # Box too large
    pred_vals[1, 3, 3, 2, 0] = -np.log(3)  # evaluates to .25
    pred_vals[1, 3, 3, 2, 1] = -np.log(3)  # evaluates to .25
    pred_vals[1, 3, 3, 2, 2] = width_inv(0.2, 2)
    pred_vals[1, 3, 3, 2, 3] = height_inv(0.2, 2)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[1, 3, 3, 2] - .1*.2*13**2) < tol
    assert np.abs(pred_area[1, 3, 3, 2] - .2*.2*13**2) < tol
    assert np.abs(inter_area[1, 3, 3, 2] - .1*.2*13**2) < tol


    # Box off-center
    pred_vals[1, 3, 3, 2, 0] = 0
    pred_vals[1, 3, 3, 2, 1] = 0
    pred_vals[1, 3, 3, 2, 2] = width_inv(0.2, 2)
    pred_vals[1, 3, 3, 2, 3] = height_inv(0.1, 2)

    true_area, pred_area, inter_area = get_areas(truth_array, pred_vals)
    assert np.abs(true_area[1, 3, 3, 2] - .1*.2*13**2) < tol
    assert np.abs(pred_area[1, 3, 3, 2] - .1*.2*13**2) < tol
    assert np.abs(inter_area[1, 3, 3, 2] - 1.05*2.35) < tol


def set_pred_vals(
        xy_vals, x_val, y_val,
        wh_vals, width, height,
        sample, grid_y, grid_x, anchor_ind):
    # Prevent random boxes being best
    xy_vals[sample, grid_y, grid_x, :, :] = 0
    wh_vals[sample, grid_y, grid_x, :, :] = 0

    xy_vals[sample, grid_y, grid_x, anchor_ind, 0] = x_val
    xy_vals[sample, grid_y, grid_x, anchor_ind, 1] = y_val

    wh_vals[sample, grid_y, grid_x, anchor_ind, 0] = np.sqrt(width)
    wh_vals[sample, grid_y, grid_x, anchor_ind, 1] = np.sqrt(height)


def test_localization_loss(loss):
    tol = 1e-4
    loss_coeff = 5

    ground_truth = np.array([[1, .5, .5, .5, .5],
                             [5, .25, .25, .2, .1],
                             [1, .5, .5, .5, .5, 2, .75, .75, .75, .75]])
    ground_truth, pred_vals = prepare_init_values(ground_truth)

    convert_model_outputs = evaluate_results(pytest.loss._convert_model_outputs)
    convert_truth_values = evaluate_results(pytest.loss._convert_truth_values)
    localization_loss = evaluate_results(pytest.loss._localization_loss)

    # Exact match
    pred_box_xy, pred_box_wh, _, _ = convert_model_outputs(pred_vals)
    set_pred_vals(
        pred_box_xy, .5, .5,  # xy vals
        pred_box_wh, .5, .5,  # wh vals
        0, 6, 6, 2)  # sample, grid cell and anchor box
    true_box_xy, true_box_wh, true_box_conf, _ = \
        convert_truth_values(ground_truth, pred_box_xy, pred_box_wh)

    loc_loss = pytest.loss._localization_loss(
        true_box_conf[0:1, :],
        true_box_xy[0:1, :], true_box_wh[0:1, :],
        pred_box_xy[0:1, :], pred_box_wh[0:1, :])
    loc_loss = loc_loss.eval()
    assert loc_loss == 0


    # Wrong grid cell, some overlap with neighbor
    pred_box_xy, pred_box_wh, _, _ = convert_model_outputs(pred_vals)
    set_pred_vals(
        pred_box_xy, 0, 0, # xy vals
        pred_box_wh, 0, 0, # wh vals
        1, 3, 3, 2) # sample, grid cell and anchor box
    set_pred_vals(
        pred_box_xy, .25, .25, # xy vals
        pred_box_wh, .5, .5, # wh vals
        1, 3, 4, 2) # sample, grid cell and anchor box

    true_box_xy, true_box_wh, true_box_conf, _ = \
        convert_truth_values(ground_truth, pred_box_xy, pred_box_wh)

    loc_loss = localization_loss(
        true_box_conf[1:2, :],
        true_box_xy[1:2, :], true_box_wh[1:2, :],
        pred_box_xy[1:2, :], pred_box_wh[1:2, :])
    assert np.isclose(loc_loss, loss_coeff*(
                          true_box_xy[1, 3, 3, 2, 0]**2
                          + true_box_xy[1, 3, 3, 2, 1]**2
                          + true_box_wh[1, 3, 3, 2, 0]**2
                          + true_box_wh[1, 3, 3, 2, 1]**2),
                      atol=tol)


    # Off-center
    pred_box_xy, pred_box_wh, _, _ = convert_model_outputs(pred_vals)
    set_pred_vals(
        pred_box_xy, .25, .25, # xy vals
        pred_box_wh, .5, .5, # wh vals
        0, 6, 6, 2) # sample, grid cell and anchor box
    true_box_xy, true_box_wh, true_box_conf, _ = \
        convert_truth_values(ground_truth, pred_box_xy, pred_box_wh)

    loc_loss = localization_loss(
        true_box_conf[0:1, :],
        true_box_xy[0:1, :], true_box_wh[0:1, :],
        pred_box_xy[0:1, :], pred_box_wh[0:1, :])
    assert np.isclose(loc_loss, loss_coeff*(.25**2 + .25**2), atol=tol)


    # Wrong size
    pred_box_xy, pred_box_wh, _, _ = convert_model_outputs(pred_vals)
    set_pred_vals(
        pred_box_xy, .5, .5, # xy vals
        pred_box_wh, .4, .4, # wh vals
        0, 6, 6, 2) # sample, grid cell and anchor box
    true_box_xy, true_box_wh, true_box_conf, _ = \
        convert_truth_values(ground_truth, pred_box_xy, pred_box_wh)

    loc_loss = localization_loss(
        true_box_conf[0:1, :],
        true_box_xy[0:1, :], true_box_wh[0:1, :],
        pred_box_xy[0:1, :], pred_box_wh[0:1, :])
    assert np.isclose(loc_loss,
                      loss_coeff*(2*(np.sqrt(.4) - np.sqrt(.5))**2),
                      atol=tol)


    # Multiple objects, box wrong size and off-center
    pred_box_xy, pred_box_wh, _, _ = convert_model_outputs(pred_vals)
    set_pred_vals(
        pred_box_xy, .25, .25, # xy vals
        pred_box_wh, .4, .4, # wh vals
        2, 6, 6, 2) # sample, grid cell and anchor box
    set_pred_vals(
        pred_box_xy, .5, .5, # xy vals
        pred_box_wh, .6, .6, # wh vals
        2, 9, 9, 2) # sample, grid cell and anchor box
    true_box_xy, true_box_wh, true_box_conf, _ = \
        convert_truth_values(ground_truth, pred_box_xy, pred_box_wh)

    loc_loss = localization_loss(
        true_box_conf[2:3, :],
        true_box_xy[2:3, :], true_box_wh[2:3, :],
        pred_box_xy[2:3, :], pred_box_wh[2:3, :])
    assert np.isclose(loc_loss,
                      loss_coeff*(
                          2*(.25)**2
                          + 2*(.25)**2
                          + 2*(np.sqrt(.4) - np.sqrt(.5))**2
                          + 2*(np.sqrt(.6) - np.sqrt(.75))**2),
                      atol=tol)


    # Multiple examples
    pred_box_xy, pred_box_wh, _, _ = convert_model_outputs(pred_vals)
    set_pred_vals(
        pred_box_xy, .25, .25, # xy vals
        pred_box_wh, .4, .4, # wh vals
        2, 6, 6, 2) # sample, grid cell and anchor box
    set_pred_vals(
        pred_box_xy, .5, .5, # xy vals
        pred_box_wh, .6, .6, # wh vals
        2, 9, 9, 2) # sample, grid cell and anchor box
    set_pred_vals(
        pred_box_xy, 0, 0, # xy vals
        pred_box_wh, 0, 0, # wh vals
        1, 3, 3, 2) # sample, grid cell and anchor box
    set_pred_vals(
        pred_box_xy, .25, .25, # xy vals
        pred_box_wh, .5, .5, # wh vals
        1, 3, 4, 2) # sample, grid cell and anchor box
    true_box_xy, true_box_wh, true_box_conf, _ = \
        convert_truth_values(ground_truth, pred_box_xy, pred_box_wh)

    loc_loss = localization_loss(
        true_box_conf[1:3, :],
        true_box_xy[1:3, :], true_box_wh[1:3, :],
        pred_box_xy[1:3, :], pred_box_wh[1:3, :])
    correct_loss = loss_coeff*(2*(.25)**2 + 2*(.25)**2
                              + 2*(np.sqrt(.4) - np.sqrt(.5))**2
                              + 2*(np.sqrt(.6) - np.sqrt(.75))**2
                              + true_box_xy[1, 3, 3, 2, 0]**2
                              + true_box_xy[1, 3, 3, 2, 1]**2
                              + true_box_wh[1, 3, 3, 2, 0]**2
                              + true_box_wh[1, 3, 3, 2, 1]**2)
    assert np.isclose(loc_loss, correct_loss, atol=tol)


def test_object_confidence_loss(loss):
    confidence_loss = evaluate_results(pytest.loss._object_confidence_loss)
    grid_w, grid_h = pytest.loss.shape[1:3]
    noobj_loss_coeff = .5
    obj_loss_coeff = 1
    tol = 1e-4

    true_conf = np.zeros((4, grid_w, grid_h, pytest.loss.n_anchors, 1))
    true_conf[0, 5, 7, 2, 0] = 1
    true_conf[1, 9, 4, 4, 0] = 1
    true_conf[2, 8, 1, 0, 0] = 1
    true_conf[2, 1, 8, 3, 0] = 1


    # Exact match
    pred_vals = np.zeros((1, grid_w, grid_h, pytest.loss.n_anchors, 1))
    pred_vals[0, 5, 7, 2, 0] = 1

    conf_loss = pytest.loss._object_confidence_loss(
        true_conf[0:1, :], pred_vals)
    conf_loss = conf_loss.eval()
    assert conf_loss == 0


    # Half confidence
    pred_vals = np.zeros((1, grid_w, grid_h, pytest.loss.n_anchors, 1))
    pred_vals[0, 5, 7, 2, 0] = .5

    conf_loss = confidence_loss(true_conf[0:1, :], pred_vals)
    assert np.isclose(conf_loss, obj_loss_coeff*.5**2, atol=tol)


    # Multiple wrong guesses and half-confident correct
    pred_vals = np.zeros((1, grid_w, grid_h, pytest.loss.n_anchors, 1))
    pred_vals[0, 9, 4, 3, 0] = .75
    pred_vals[0, 9, 5, 4, 0] = 1
    pred_vals[0, 9, 4, 4, 0] = .5

    conf_loss = confidence_loss(true_conf[1:2, :], pred_vals)
    assert np.isclose(conf_loss,
                      noobj_loss_coeff*(.75**2 + 1) + obj_loss_coeff*(.5**2),
                      atol=tol)


    # Multiple objects present
    pred_vals = np.zeros((1, grid_w, grid_h, pytest.loss.n_anchors, 1))
    pred_vals[0, 8, 1, 0, 0] = 1
    pred_vals[0, 1, 8, 3, 0] = .5
    pred_vals[0, 8, 8, 4, 0] = 1

    conf_loss = confidence_loss(true_conf[2:3, :], pred_vals)
    assert np.isclose(conf_loss,
                      noobj_loss_coeff*(1) + obj_loss_coeff*(.5**2),
                      atol=tol)


    # Multiple samples
    pred_vals = np.zeros((2, grid_w, grid_h, pytest.loss.n_anchors, 1))
    pred_vals[0, 5, 7, 2, 0] = .75
    pred_vals[1, 9, 4, 4, 0] = .5
    pred_vals[0, 0, 0, 4, 0] = 1
    pred_vals[1, 0, 0, 4, 0] = 1

    conf_loss = confidence_loss(true_conf[0:2, :], pred_vals)
    assert np.isclose(conf_loss,
                      noobj_loss_coeff*(2*1) + obj_loss_coeff*(.5**2 + .25**2),
                      atol=tol)


def test_category_loss(loss):
    category_loss = evaluate_results(pytest.loss._category_loss)
    grid_w, grid_h = pytest.loss.shape[1:3]
    loss_coeff = 1.0
    tol = 1e-4

    true_prob = np.zeros((3,
                          grid_w, grid_h,
                          pytest.loss.n_anchors,
                          pytest.loss.n_classes))
    true_conf = np.zeros((3,
                          grid_w, grid_h,
                          pytest.loss.n_anchors,
                          1))
    true_prob[0, 5, 7, :, 0] = 1
    true_conf[0, 5, 7, 2, 0] = 1

    true_prob[1, 9, 4, :, 1] = 1
    true_conf[1, 9, 4, 1, 0] = 1

    true_prob[2, 8, 1, :, 2] = 1
    true_prob[2, 1, 8, :, 3] = 1
    true_conf[2, 8, 1, 2, 0] = 1
    true_conf[2, 1, 8, 3, 0] = 1


    # Exact match
    pred_prob = np.random.random((1,
                                  grid_w, grid_h,
                                  pytest.loss.n_anchors,
                                  pytest.loss.n_classes))
    pred_prob[0, 5, 7, 2, :] = 0
    pred_prob[0, 5, 7, 2, 0] = 1
    prob_loss = pytest.loss._category_loss(true_conf[0:1, :],
                                           true_prob[0:1, :],
                                           pred_prob)
    prob_loss = prob_loss.eval()
    assert prob_loss == 0


    # Partial match
    pred_prob = np.random.random((1,
                                  grid_w, grid_h,
                                  pytest.loss.n_anchors,
                                  pytest.loss.n_classes))
    pred_prob[0, 9, 4, 1, :] = 0
    pred_prob[0, 9, 4, 1, 0] = .5
    pred_prob[0, 9, 4, 1, 1] = .5
    prob_loss = category_loss(true_conf[1:2, :], true_prob[1:2, :], pred_prob)
    assert np.isclose(prob_loss,
                      loss_coeff*(.5**2 + .5**2),
                      atol=tol)


    # Multi-object partial match
    pred_prob = np.random.random((1,
                                  grid_w, grid_h,
                                  pytest.loss.n_anchors,
                                  pytest.loss.n_classes))
    pred_prob[0, 8, 1, 2, :] = 0
    pred_prob[0, 1, 8, 3, :] = 0
    pred_prob[0, 8, 1, 2, 1] = .25
    pred_prob[0, 8, 1, 2, 2] = .75
    pred_prob[0, 1, 8, 3, 3] = 1
    prob_loss = category_loss(true_conf[2:3, :], true_prob[2:3, :], pred_prob)
    assert np.isclose(prob_loss,
                      loss_coeff*(.25**2 + .25**2),
                      atol=tol)


    # Multi-sample partial match
    pred_prob = np.random.random((2,
                                  grid_w, grid_h,
                                  pytest.loss.n_anchors,
                                  pytest.loss.n_classes))
    pred_prob[0, 5, 7, 2, :] = 0
    pred_prob[1, 9, 4, 1, :] = 0

    pred_prob[0, 5, 7, 2, 0] = 1
    pred_prob[1, 9, 4, 1, 0] = .5
    pred_prob[1, 9, 4, 1, 1] = .5
    prob_loss = category_loss(true_conf[0:2, :], true_prob[0:2, :], pred_prob)
    assert np.isclose(prob_loss,
                      loss_coeff*(.5**2 + .5**2),
                      atol=tol)


def test_loss(loss):
    """
    Test full loss function

    Mostly intended to check input handling,
    previous tests check accuracy.
    """
    loss = evaluate_results(pytest.loss.loss)
    grid_w, grid_h = pytest.loss.shape[1:3]
    tol = 1e-5

    ground_truth = np.array([[1, .5, .5, .5, .5],
                             [5, .25, .25, .2, .1],
                             [1, .5, .5, .5, .5, 2, .75, .75, .75, .75]])
    truth_array, pred_vals = prepare_init_values(ground_truth)

    def width_inv(target_w, anchor_ind):
        pred_w = np.log((target_w*grid_w)/pytest.loss.anchors[anchor_ind][0])
        return pred_w

    def height_inv(target_h, anchor_ind):
        pred_h = np.log((target_h*grid_h)/pytest.loss.anchors[anchor_ind][1])
        return pred_h

    # Zero confidence values (zeroed after conversion)
    pred_vals[:, :, :, :, 4] = -10000


    # Exact match
    categories = np.zeros((pytest.loss.n_classes,))
    categories[0] = 10000
    pred_vals[0, 6, 6, 0, 0] = 0  # x_vals - denotes box center
    pred_vals[0, 6, 6, 0, 1] = 0  # y_vals - denotes box center
    pred_vals[0, 6, 6, 0, 2] = width_inv(0.5, 0)
    pred_vals[0, 6, 6, 0, 3] = height_inv(0.5, 0)
    pred_vals[0, 6, 6, 0, 4] = 1000
    pred_vals[0, 6, 6, 0, 5:] = categories

    tot_loss = pytest.loss.loss(truth_array[0:1, :], pred_vals[0:1, :])
    tot_loss = tot_loss.eval()
    assert np.isclose(tot_loss, 0, atol=tol)


    # Exact match with multiple samples
    categories = np.zeros((pytest.loss.n_classes,))
    categories[4] = 10000
    pred_vals[1, 3, 3, 2, 0] = -np.log(3)
    pred_vals[1, 3, 3, 2, 1] = -np.log(3)
    pred_vals[1, 3, 3, 2, 2] = width_inv(0.2, 2)
    pred_vals[1, 3, 3, 2, 3] = height_inv(0.1, 2)
    pred_vals[1, 3, 3, 2, 4] = 1000
    pred_vals[1, 3, 3, 2, 5:] = categories

    tot_loss = loss(truth_array[0:2, :], pred_vals[0:2, :])
    assert np.isclose(tot_loss, 0, atol=tol)


    # Exact match with multiple samples and objects
    categories = np.zeros((pytest.loss.n_classes,))
    categories[0] = 10000
    pred_vals[2, 6, 6, 1, 0] = 0
    pred_vals[2, 6, 6, 1, 1] = 0
    pred_vals[2, 6, 6, 1, 2] = width_inv(0.5, 1)
    pred_vals[2, 6, 6, 1, 3] = height_inv(0.5, 1)
    pred_vals[2, 6, 6, 1, 4] = 1000
    pred_vals[2, 6, 6, 1, 5:] = categories

    categories = np.zeros((pytest.loss.n_classes,))
    categories[1] = 10000
    pred_vals[2, 9, 9, 3, 0] = -np.log(1/3)
    pred_vals[2, 9, 9, 3, 1] = -np.log(1/3)
    pred_vals[2, 9, 9, 3, 2] = width_inv(0.75, 3)
    pred_vals[2, 9, 9, 3, 3] = height_inv(0.75, 3)
    pred_vals[2, 9, 9, 3, 4] = 1000
    pred_vals[2, 9, 9, 3, 5:] = categories

    tot_loss = loss(truth_array[2:3, :], pred_vals[2:3, :])
    assert np.isclose(tot_loss, 0, atol=tol)
