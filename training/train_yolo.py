"""
This file provides functions used to modify and
train a Keras representation of the YOLO v2 neural net.

The output of the model described in the Direct Local Prediction
section of: https://arxiv.org/pdf/1612.08242.pdf
"""

from keras.layers import InputLayer, Conv2D
from keras import models
import numpy as np
from skimage.transform import rescale

from utils.io import Labels
from yolo_loss import YoloLoss


def build_arg_dict(arg_list):
    arg_dict = {
        'model_file': arg_list.model_file,
        'bbox_file': arg_list.bbox_file,
        'image_dir': arg_list.image_dir,
    }
    return arg_dict


class YoloModel:
    def __init__(self, model_file, anchor_file, input_dim, n_classes):
        self.anchors = self._read_anchor_boxes(anchor_file)
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        self.input_dim = input_dim

        # Prepare the model. Its input can later be resized
        self.model = models.load_model(model_file)
        self._prep_model()

    @staticmethod
    def _read_anchor_boxes(anchor_file):
        """
        Gets anchor box definitions.

        Anchor box dimensions should be listed as fraction of image
        dimensions.

        :param string anchor_file: Path to the anchor box file.
        :return: anchors as a numpy array
        """
        with open(anchor_file, 'r') as fanchors:
            anchors = []
            for anchor in fanchors:
                w, h = anchor.split(',')
            anchors.append((float(w), float(h)))
        return np.array(anchors)

    def _prep_model(self):
        """
        Create a model capable of making predictions on the given
        number of classes

        :param model_file: File to a pretrained YOLO v2 model
        :return: The prepared model
        """
        self.resize(self.input_dim)

        # Output dimension changes based on number of classes predicted
        out_layer = Conv2D(self.n_anchors * (5 + self.n_classes), 1)
        out = out_layer(self.model.layers[-2].get_output)
        self.model = models.Model(model.get_input_at(-1), out)

    def set_train_status(self, status, out_matches=False):
        """
        Sets the model's layers to be frozen or unfrozen according to status

        :param bool status: If True model's layers set to trainable
        :param out_matches: Whether output status should match other layers
        """
        # Seems unnecessarily general
        for layer in self.model.layers:
            layer.trainable = status
        if not out_matches:
            self.model.layers[-1].trainable = not status

    def train(self, train_labels, valid_labels, epochs, file_label=''):
        """
        Trains the model for a given number of epochs on the data
        referenced by labels.

        :param train_labels: training data. Labels object set to proc_image
        :param valid_labels: validation data. Labels object set to proc_image
        :param epochs: Number of epochs to train for
        :return history: Keras history object
        """
        if not file_label:
            file_label = self.model.name
        checkpointer = ModelCheckpoint(
            filepath='model_data/{}.hdf5'.format(file_label),
            verbose=1, save_best_only=True)
        tb = keras.callbacks.TensorBoard(
            log_dir='model_data/{}_tb'.format(file_label),
            histogram_freq=0,
            write_graph=True, write_images=True)

        rescaled_train = rescaled_image_gen(train_labels, self.input_dim,
                                            self.n_classes, self.n_anchors)
        rescaled_valid = rescaled_image_gen(valid_labels, self.input_dim,
                                            self.n_classes, self.n_anchors)

        history = model.fit_generator(rescaled_train,
                                      steps_per_epoch=train_labels.n_chunk,
                                      epochs=epochs,
                                      verbose=1,
                                      callbacks=[checkpointer, tb],
                                      validation_data=rescaled_valid,
                                      validation_steps=valid_labels.n_chunk)
        return history

    def resize(self, new_input_dim):
        """
        Change model to accept input_shape images as input

        :param new_input_dim: The new input dimensions for the model to accept
        :return: model that accepts input of dimension input_shape
        """
        inp = InputLayer(new_input_dim)
        self.model(inp.output)
        self.model = models.Model(inp.input, self.model.get_output_at(-1))
        self.input_dim = new_input_dim
        self._compile_model()

    def _compile_model(self):
        """
        Couples the model to its loss function.

        :param n_classes:
        :param input_dim:
        :return:
        """
        out_dim = self.input_dim // 32  # factor of 32 decrease in input
        yolo_loss = YoloLoss(self.anchors * out_dim, self.n_classes, out_dim)
        model.compile('adam', loss=yolo_loss)


def expand_truth_vals(ground_truth, n_classes, grid_dims, n_anchors):
    """
    Converts a ground truth entry to an array that can be used to evaluate
    model output.

    Need to rework loss function calculation to handle boxes that
    share a center grid cell.

    :param ground_truth: (n_objects x 6) numpy array of floats.
        Six entries are:
           - Image index (0-indexed)
           - category number - 1-indexed
           - x, y coord bbox center - \in [0, 1]
           - bbox width in x, y - \in [0, 1]
    :return: truth_array:
        Five dimensions corresponding to:
            - Number of samples
            - grid rows, columns
            - anchor box priors
            - model predicted values (center, dimensions,
              confidence, classes)
    """
    n_samples = ground_truth.shape[0]
    ground_truth = np.array([[i, *x]
                             for i, s in enumerate(ground_truth)
                             for x in np.reshape(s, (-1, 5))])
    sample, cat, cx, cy, wx, wy = ground_truth.T
    n_objects = sample.shape[0]

    one_hot = np.zeros((n_objects, n_classes))
    one_hot[range(n_objects), cat.astype(np.int32) - 1] = 1
    wx = np.sqrt(wx)
    wy = np.sqrt(wy)

    # TODO Check grid row and column treated correctly
    object_mask = np.concatenate(
        [np.array([cx, cy, wx, wy, np.ones(n_objects)]).T, one_hot],
        axis=1)
    center_x = cx * grid_dims[0]
    center_y = cy * grid_dims[1]
    r = np.floor(center_x).astype(np.int32)
    c = np.floor(center_y).astype(np.int32)

    # Loss only calculated if object lands in anchor box
    truth_array = np.zeros((n_samples,
                            grid_dims[0], grid_dims[1],
                            n_anchors, n_classes + 5))
    truth_array[sample.astype(np.int32), c, r, :, :] = np.reshape(
        object_mask,
        (n_objects, 1, n_classes + 5))
    return truth_array.astype(np.float32)


def rescaled_image_gen(labels, target_dim, n_classes, n_anchors):
    """
    Scales all square images returned from a Labels object to be
    target_dim x target_dim in size

    :param labels: A labels object set to target proc_image
    :param target_dim: The desired image dimension
    :returns: image and bounding box info
    """
    while True:
        chunk = next(labels)
        cur_shape = chunk['data'].values[0].shape
        chunk['data'] = chunk['data'].apply(rescale,
                                            scale=target_dim/cur_shape[0])

        truth_vals = expand_truth_vals(chunk['category_label',
                                             'center_x', 'center_y',
                                             'width_x', 'width_y'].values,
                                       n_classes, target_dim/32, n_anchors)
        yield np.stack(chunk['data'].values), truth_vals


def stratified_train_val_test(y):
    """Generate stratified train-val-test splits. Hard coded for 70-20-10 proportion"""
    np.random.seed(285)
    # Train-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1)
    train, test = next(sss.split(y, y))

    # Train-validate split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=2/9)
    train, valid = next(sss.split(y[train], y[train]))
    return train, valid, test


def subset_labels(indices, labels, chunksize, name):
    sub_labels = Labels(labels.labels.iloc[indices, :],
                        labels.image_path_prefix)
    sub_labels.set_data_target('proc_image', chunksize, name)
    return sub_labels


def train_yolo(arg_dict):
    yolo_model = YoloModel(arg_dict['model_file'],
                           arg_dict['anchor_file'],
                           input_dim=(288, 288, 3),
                           n_classes=50)

    chunksize = 3000
    labels = Labels(arg_dict['bbox_file'], arg_dict['image_dir'],
                    n_images_loaded=50)
    train, valid, test = stratified_train_val_test(
        labels.labels['category_label'].values)
    train_labels = subset_labels(train, labels, chunksize, '')
    valid_labels = subset_labels(valid, labels, chunksize, '')

    # Train output layer with smallest considered image dimension
    yolo_model.set_train_status(False, out_matches=False)
    yolo_model.train(train_labels, valid_labels, epochs=500, file_label='init')

    # Dimensions copied from YOLO 9000 paper
    input_dims = list(range(320, 609, 64))
    for _ in range(20):
        inp = choice(input_dims)
        yolo_model.resize((inp, inp, 3))
        yolo_model.train(train_labels, valid_labels, epochs=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the YOLO v2 model at multiple length scales')
    parser.add_argument(
        'model_file', nargs='?',
        default='yolo_v2/YAD2K/model_data/yolo.h5',
        help='Path to the pre-trained YOLO v2 model')
    parser.add_argument(
        'bbox_file', nargs='?',
        default='data/Category and Attribute Prediction Benchmark/Anno/yolo_list_bbox.txt',
        help='File containing mapping images to bounding box within image')
    parser.add_argument(
        'image_dir', nargs='?',
        default='data/Category and Attribute Prediction Benchmark/Img/Img/',
        help='Directory containing image references')
    args = parser.parse_args()

    arg_dict = build_arg_dict(args)
    convert_bbox(arg_dict)
