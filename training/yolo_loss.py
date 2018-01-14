"""
All functions used to define the YOLO loss function

"""

import keras.backend as K
import tensorflow as tf
import numpy as np
import pdb


class YoloLoss:
    """
    Loss Function of YOLOv2.

    https://github.com/datlife/yolov2 yolo_loss function used as base.
    """
    def __init__(self, anchors, n_classes, grid_dims, batch_size=32):
        """
        Preps values used throughout the loss calculation for a given input
        image dimension.

        :param anchors: A series of width and height tuples.
                        width and height \in [0, grid_dims].
        :param n_classes: The number of classes being predicted.
        :param grid_dims: The width and height of the grid classifications
                          performed on.
        """
        self.anchors = anchors
        self.batch_size = batch_size

        # Output dimensions
        self.grid_h = tf.cast(grid_dims[0], tf.int32)
        self.grid_w = tf.cast(grid_dims[1], tf.int32)
        self.n_anchors = len(self.anchors)
        self.n_classes = n_classes
        self.shape = (-1,
                      grid_dims[0],
                      grid_dims[1],
                      self.n_anchors,
                      self.n_classes + 5)

        # Needed for conversion
        self.output_size = tf.cast(tf.reshape(
            [self.grid_w, self.grid_h],
            [1, 1, 1, 1, 2]), tf.float32)
        self.c_xy = self._create_offset_map()

        # Do here to avoid overhead from repeated variable creation
        self.truth_array = tf.get_variable("truth_array",
                                           (self.batch_size,
                                            grid_dims[0], grid_dims[1],
                                            self.n_anchors, self.n_classes + 5),
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer)

    def _convert_model_outputs(self, y_pred):
        """
        Takes values predicted by model and converts them to more interpretable
        values.

        :param y_pred: The raw model output
        :return pred_box_xy: Predicted xy of bounding box center.
                             Does not use fractional representation.
        :return pred_box_wh: Square root of predicted bounding box dimensions.
                             wh converted to fractional rep before square root
        :return pred_box_conf: Confidence an object is in box
        :return pred_box_prob: The probability of an object belonging to a class
        """
        pred_box_xy = (tf.sigmoid(y_pred[:, :, :, :, :2]) + self.c_xy) / self.output_size
        pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) \
                    * np.reshape(self.anchors, [1, 1, 1, self.n_anchors, 2]) \
                    / self.output_size
        pred_box_wh = tf.sqrt(pred_box_wh)
        pred_box_conf = tf.sigmoid(y_pred[:, :, :, :, 4:5])
        pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
        return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob

    def _convert_truth_values(self, y_true, pred_box_xy, pred_box_wh):
        """
        Breaks up the y_true array. Only work done is zeroing the
        true_box_conf entries for all but the best anchor box.

        :param y_true: The ground truth array
        :param pred_box_area: The area of the predicted bounding box
        :return true_box_conf: Whether an object is present in a grid cell
        :return true_box_prob: What category the present object belongs to
        """
        true_box_xy = y_true[:, :, :, :, 0:2]
        true_box_wh = y_true[:, :, :, :, 2:4]
        true_box_conf = self._calc_true_confidence(
            y_true[:, :, :, :, 4],
            true_box_xy, true_box_wh,
            pred_box_xy, pred_box_wh)
        true_box_prob = y_true[:, :, :, :, 5:]
        return true_box_xy, true_box_wh, true_box_conf, true_box_prob

    def _calc_true_confidence(self, orig_true_conf, true_box_xy, true_box_wh,
                              pred_box_xy, pred_box_wh):
        # Highest IoU indicates the best anchor box for each cell.
        # Each grid cell can only contain one object center.
        true_box_area, pred_box_area, intersect_area = self._calc_all_areas(
            true_box_xy, true_box_wh,
            pred_box_xy, pred_box_wh)
        iou = tf.truediv(intersect_area,
                         true_box_area + pred_box_area - intersect_area)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], keep_dims=True))
        best_box = tf.to_float(best_box)

        # Get the relevant object confidence and category probabilities
        true_box_conf = tf.expand_dims(best_box * orig_true_conf, -1)
        return true_box_conf

    @staticmethod
    def _calc_box_stats(box_xy, box_wh, output_size):
        """
        Calculates coordinates for the upper left and bottom right box corners
        as well as the box area.

        :param box_xy: xy values for center of a bounding box.
                       xy values not in fractional representation.
        :param box_wh: sqrt width and height of true bounding box.
                       wh in fractional representation before
                       square root applied.
        :param output_size: Array containing number of grid cells in the
                            x and y directions.
        :return: box_ul, box_br, box_area
        """
        tem_wh = tf.pow(box_wh, 2) * output_size
        box_ul = (box_xy - 0.5 * tf.pow(box_wh, 2)) * output_size
        box_br = (box_xy + 0.5 * tf.pow(box_wh, 2)) * output_size
        box_area = tem_wh[:, :, :, :, 0] * tem_wh[:, :, :, :, 1]
        return box_ul, box_br, box_area

    @staticmethod
    def _calc_intersect_area(pred_box_ul, pred_box_br, true_box_ul, true_box_br):
        """
        Calculates the intersection between predicted and true bounding boxes

        :param pred_box_ul: The xy values for the upper left corner of the
                            predicted bounding box.
                            xy values not in fractional representation.
        :param pred_box_br: Same but for bottom right.
        :param true_box_ul: Analogous to predicted.
        :param true_box_br: Analogous to predicted.
        :return intersect_area:
        """
        intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
        intersect_br = tf.minimum(pred_box_br, true_box_br)
        intersect_wh = tf.maximum(intersect_br - intersect_ul, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        return intersect_area

    def _calc_all_areas(self,
                        true_box_xy, true_box_wh,
                        pred_box_xy, pred_box_wh):
        """
        Calculates the areas for true bounding boxes, predicted bounding boxes
        and their intersection.

        :param true_box_xy: xy values for center of the true bounding box.
                            xy values not in fractional representation.
        :param true_box_wh: sqrt width and height of true bounding box.
                            wh in fractional representation before
                            square root applied.
        :param pred_box_xy: Predicted xy. Details same as true_box_xy.
        :param pred_box_wh: Predicted wh. Details same as true_box_wh.
        :return: true_box_area, pred_box_area, intersect_area
        """
        true_box_ul, true_box_br, true_box_area = \
            self._calc_box_stats(true_box_xy, true_box_wh, self.output_size)
        pred_box_ul, pred_box_br, pred_box_area = \
            self._calc_box_stats(pred_box_xy, pred_box_wh, self.output_size)
        intersect_area = self._calc_intersect_area(
            pred_box_ul, pred_box_br,
            true_box_ul, true_box_br)
        return true_box_area, pred_box_area, intersect_area

    def _create_offset_map(self):
        """
        In Yolo9000 paper, Grid map to calculate offsets for each cell in
        the output feature map
        """
        cx = tf.cast((K.arange(0, stop=self.grid_w)), dtype=tf.float32)
        cx = K.tile(cx, [self.grid_h])
        cx = K.reshape(cx, [-1, self.grid_h, self.grid_w, 1])

        cy = K.cast((K.arange(0, stop=self.grid_h)), dtype=tf.float32)
        cy = K.reshape(cy, [-1, 1])
        cy = K.tile(cy, [1, self.grid_w])
        cy = K.reshape(cy, [-1])
        cy = K.reshape(cy, [-1, self.grid_h, self.grid_w, 1])

        c_xy = tf.stack([cx, cy], -1)
        c_xy = K.cast(c_xy, tf.float32)
        return c_xy

    def _localization_loss(self, true_box_conf,
                           true_box_xy, true_box_wh,
                           pred_box_xy, pred_box_wh):
        """
        Calculates the loss due to localization error

        Formula: mean_{over examples}(\sum_{all grid boxes and best anchor box}
                        [{(\sigma(t_x) + c_x)/x_grid_len - box_x}^2
                      +  {(\sigma(t_y) + c_y)/y_grid_len - box_y}^2
                      +  {sqrt(p_w e^(t_w)) - sqrt(box_w)}^2
                      +  {sqrt(p_h e^(t_h)) - sqrt(box_h)}^2]*5*box_conf

        :param true_box_xy:
        :param true_box_wh:
        :param pred_box_xy:
        :param pred_box_wh:
        :return: loc_loss
        """
        loss_coeff = 5.0
        # Average over anchor boxes with matching IoU. Very rare, include?
        anchor_averaging = tf.equal(true_box_conf, tf.reduce_max(true_box_conf, 3, keep_dims=True))
        anchor_averaging = tf.to_float(anchor_averaging)
        true_box_conf = true_box_conf/tf.reduce_sum(anchor_averaging, 3, keep_dims=True)

        weight_coor = loss_coeff * tf.concat(4 * [true_box_conf], 4)
        true_boxes = tf.concat([true_box_xy, true_box_wh], 4)
        pred_boxes = tf.concat([pred_box_xy, pred_box_wh], 4)

        loc_loss = tf.pow(true_boxes - pred_boxes, 2) * weight_coor
        loc_loss = tf.reshape(loc_loss, [-1, self.grid_w * self.grid_h
                                        * self.n_anchors * 4])

        # Average over samples not objects
        # Average not specified in YOLO paper turn into sum
        loc_loss = tf.reduce_sum(loc_loss)
        return loc_loss

    def _object_confidence_loss(self, true_box_conf, pred_box_conf):
        """
        Loss due to error in predicting presence or non-presence of an object

        Formula: mean_{over examples}(\sum_{all grid boxes and anchors}
                        [true_box_conf      *{true_conf_grid_box_i - pred_conf_grid_box_i}^2
                    + .5*(1 - true_box_conf)*{true_conf_grid_box_i - pred_conf_grid_box_i}^2]

        :param true_box_conf: Whether an object is in a given anchor box
        :param pred_box_conf: Model confidence an anchor box contains an object
        :return: obj_conf_loss
        """
        noobj_coeff = 0.5
        obj_coeff = 1.0
        # weight_conf eliminates values that don't have/have (in that order)
        # an object in them (previous implementation weighted positive higher than specified)
        weight_conf = noobj_coeff*(1. - true_box_conf) + obj_coeff*true_box_conf
        obj_conf_loss = tf.pow(true_box_conf - pred_box_conf, 2) * weight_conf
        obj_conf_loss = tf.reshape(obj_conf_loss,
                                   [-1, self.grid_w * self.grid_h * self.n_anchors])
        obj_conf_loss = tf.reduce_sum(obj_conf_loss)
        return obj_conf_loss

    def _category_loss(self, true_box_conf, true_box_prob, pred_box_prob):
        """
        Returns the loss due to classification error. Error only considered when
        a grid cell has an object in it.

        :param true_box_conf: Whether an object is in a given anchor box
        :param true_box_prob: OHE vector denoting category for a grid cell.
                              All zeros for cells without an object.
        :param pred_box_prob: Predicted category probabilities for a grid cell
        :return: category_loss
        """
        loss_coeff = 1.0
        weight_prob = loss_coeff * tf.concat(self.n_classes * [true_box_conf], 4)
        category_loss = tf.pow(true_box_prob - pred_box_prob, 2) * weight_prob
        category_loss = tf.reshape(category_loss,
                                   [-1, self.grid_w * self.grid_h
                                                    * self.n_anchors
                                                    * self.n_classes])
        category_loss = tf.reduce_sum(category_loss)
        return category_loss

    def loss(self, y_true, y_pred):
        """
        Calculates loss for the YOLO model.

        Loss is not averaged over samples or objects. Control
        for this when comparing loss on different image sets.

        :param y_true: 2-D list of target values, defined as follows:
            First dimension indexes images,
            Second dimension describes objects. Each object described
            by five numbers. In order:
                - (1-indexed) category label
                - x-coord of box center
                - y-coord of box center
                - box width
                - box height
            These five numbers repeat for each object in the image.
            The x-coord, y-coord, width and height should be represented as a
            fraction of the box dimensions (be between 0 and 1).
            The xy coordinate system has its origin at the top left of
            the image. Moving down increases y (makes it positive), moving
            right increases x.

        :param y_pred: Model output values.
        :return: loss
        """
        # Loss function defined in original YOLO paper: https://arxiv.org/pdf/1506.02640.pdf
        # Modify so that 1_ij instead of 1_i used in last term
        shape = [-1, self.grid_w, self.grid_h, self.n_anchors, self.n_classes+5]
        y_pred = tf.reshape(y_pred, shape)

        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob = \
                     self._convert_model_outputs(y_pred)
        true_box_xy, true_box_wh, true_box_conf, true_box_prob = \
                     self._convert_truth_values(y_true, pred_box_xy, pred_box_wh)

        loc_loss = self._localization_loss(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
        obj_conf_loss = self._object_confidence_loss(true_box_conf, pred_box_conf)
        category_loss = self._category_loss(true_box_conf, true_box_prob, pred_box_prob)

        loss = loc_loss + obj_conf_loss + category_loss
        return loss
