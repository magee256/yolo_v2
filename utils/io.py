"""
Classes that handle reading and writing data in chunks.
"""
from skimage.io import imread  # uses PIL on the backend
from skimage.io import imsave

import numpy as np
import pandas as pd
from functools import partial

# Needed to apply preprocessing to labels. 
# I don't like introducing model dependent processing to Labels.
# Consider simplifying __next__ and transferring responsibility elsewhere
from keras.applications.resnet50 import preprocess_input as res_preproc
from keras.applications.inception_v3 import preprocess_input as inc_preproc
from keras.applications.xception import preprocess_input as xcept_preproc


class Labels:
    """
    Manages saving and loading for all data
    """
    def __init__(self, labels_info, image_path_prefix, n_images_loaded):
        self.n_top_labels = 50  # n most frequent labels
        self.image_path_prefix = image_path_prefix
        self.iter_df = True
        self.n_images_loaded = n_images_loaded
        if isinstance(labels_info, str):
            self.labels = self._determine_data_subset(labels_info)

            # Reduce image_paths to their base form
            self.labels['image_name'] = self.labels['image_name'].apply(
                lambda x: x[len('img'):-len('.jpg')])
        elif isinstance(labels_info, pd.DataFrame):
            if n_images_loaded == -1:
                self.labels = labels_info
            else:
                self.labels = labels_info.head(n_images_loaded)
        else:
            raise ValueError('Unknown data type for labels_info')

    def __len__(self):
        return len(self.labels)

    def __add__(self, other):
        if not isinstance(other, Labels):
            raise TypeError('Could not add object of type'
                            ' {} to Labels instance'.format(type(other)))
        if not self.n_images_loaded == other.n_images_loaded:
            raise ValueError('Labels objects set for differing numbers of'
                             ' images to load.')
        # Model name goes first, image variations go last
        out_labels = self.labels.append(other.labels).reset_index(drop=True)
        return Labels(out_labels, self.image_path_prefix, self.n_images_loaded)

    def __next__(self):
        """
        Returns image data and corresponding category information
        
        set_load_target must be called first. The labels 
        may be iterated over forever - the data loops upon reaching
        the end of the label list.
        """
        if self.i_chunk == self.n_chunk:
            self.i_chunk = 0
        subset = self.labels.iloc[self.i_chunk * self.chunksize:
                                  (self.i_chunk + 1) * self.chunksize, :]

        # Stop external operations affecting stored info
        subset = subset.copy()
        subset['data'] = subset['image_name'].apply(self._load_data)
        self.i_chunk += 1
        if self.iter_df:
            return subset
        else:
            # This form is most convenient when feeding to models
            subset['data'] = subset['data'].apply(self.preproc_data)
            return np.stack(subset['data'].values), \
                np.stack(subset['category_label'].values)

    def _determine_data_subset(self, labels_path):
        # Read in the attribute information to figure out what images to load
        cat_labels = pd.read_csv(labels_path, skiprows=1, sep='\s+')

        # Figure out which are the most frequent labels to keep
        cat_label_count = cat_labels.groupby('category_label').count()
        cat_label_count = cat_label_count.sort_values(
            by=['image_name'], ascending=False)
        kept_labels = cat_label_count.head(self.n_top_labels).index

        # Filter labels not in the n_top_labels most frequent
        cat_labels = cat_labels.loc[
            cat_labels['category_label'].isin(kept_labels)]

        if self.n_images_loaded != -1:
            return cat_labels.head(self.n_images_loaded)
        else:
            return cat_labels

    def one_hot_labels(self):
        """
        One hot encode category label entries, creating a map from
        their label to one hot encoded index
        """
        self.cat_map = {}
        unique_list = self.labels['category_label'].unique()
        for i, cat in enumerate(unique_list):
            self.cat_map[cat] = i

        def one_hot_encode(category):
            ohe = np.zeros(self.n_top_labels)
            ohe[self.cat_map[category]] = 1
            return ohe

        self.labels['category_label'] = self.labels['category_label'
                                                    ].apply(one_hot_encode)

    def set_data_target(self, target, chunksize, model_name=''):
        """Set the type of data, how much of it to load, and how to save it"""
        loaders = LoadMethods(self.image_path_prefix)
        load_method_dict = {
            'raw_image': loaders.load_raw_image,
            'proc_image': loaders.load_proc_image,
            'bottleneck': partial(loaders.load_bottleneck, model_name),
        }

        savers = SaveMethods(self.image_path_prefix)
        save_method_dict = {
            'raw_image': savers.save_proc_image,
            'proc_image': partial(savers.save_bottleneck, model_name),
            'bottleneck': partial(savers.save_predictions, model_name),
        }
        self._load_data = load_method_dict[target]
        self._save_data = save_method_dict[target]

        self.chunksize = chunksize
        self.i_chunk = 0
        self.n_chunk = (len(self.labels) + chunksize - 1) // chunksize
        self.load_target = model_name + ' ' + target
        self._set_preprocessing(model_name)

    def _set_preprocessing(self, model_name):
        """
        Set the type of preprocessing applied to data before it is returned by
        __next__. Only used when data target is "bottleneck" 
        """
        preproc_dict = {
            'resnet': res_preproc,
            'inception_v3': inc_preproc,
            'xception': xcept_preproc,
        }
        self.preproc_data = preproc_dict.get(model_name, lambda x: x)

    def save(self, subset_df):
        subset_df.apply(self._save_data, axis=1)


class LoadMethods():
    def __init__(self, image_path_prefix):
        self.image_path_prefix = image_path_prefix

    def load_raw_image(self, image_path):
        """Given an image path, load the data as an array"""
        load_path = self.image_path_prefix + 'img' + image_path + '.jpg'
        image = imread(load_path, plugin='pil')
        return image

    def load_proc_image(self, image_path):
        """Given an image path, load the data as an array"""
        load_path = self.image_path_prefix + 'proc_img' + image_path + '.jpg'
        image = imread(load_path, plugin='pil')
        return image

    def load_bottleneck(self, model_name, numpy_path):
        """Given a numpy path, load the data as an array"""
        load_path = self.image_path_prefix + model_name + '_bottleneck' \
                    + numpy_path + '.npy'
        bottleneck = np.load(load_path)
        return bottleneck


class SaveMethods():
    def __init__(self, image_path_prefix):
        self.image_path_prefix = image_path_prefix

    def save_proc_image(self, row):
        save_path = self.image_path_prefix + 'proc_img' + row['image_name'] + '.jpg'
        imsave(save_path, row['data'], plugin='pil')

    def save_bottleneck(self, model_name, row):
        save_path = self.image_path_prefix + model_name + '_bottleneck' \
                    + row['image_name'] + '.npy'
        np.save(save_path, row['data'])

    def save_predictions(self, model_name, row):
        pass
