# Function to read and write AutoML format
# TODO

import pandas as pd
import os

def from_automl(path):
    """ Read files in AutoML format.
        TODO
    """
    pass
    # detect files, abort if conflicts (several datasets)
    """
    feat_name_file = os.path.join(input_dir, basename + '_feat.name')
    feat_name = pd.read_csv(feat_name_file, header=None).values.ravel() if os.path.exists(feat_name_file) else None
    label_name_file = os.path.join(input_dir, basename + '_label.name')
    label_name = pd.read_csv(label_name_file, header=None).values.ravel() if os.path.exists(label_name_file) else None
    # if exists
    if os.path.exists(os.path.join(input_dir, basename + '.data')):
        # read .data and .solution
        pd.read_csv(filepath, sep=' ', header=None)
    # create AutoData object
    data = AutoData(df)
    # class ?
    data.set_class()
    # train/valid/test ?
    data.indexes['train'] = [0]
    """

def to_automl(data, path='.', name='autodata'):
    """ Write files in AutoML format.
        AutoML format is ideal to create a Codalab competition.

        :param data: AutoData frame to format.
        :param path: where to save the dataset
        :param name: name of the dataset to put into filenames
    """
    # check if folder exists
    dir = os.path.join(path, name+'_automl')
    if not os.path.exists(dir):
        # create folder
        os.mkdir(dir)
    data.descriptors().to_csv(os.path.join(dir, name+'.info'), header=True) # some information
    if data.has_class():
        pd.DataFrame(data.indexes['X']).to_csv(os.path.join(dir, name+'_feat.name'), index=False, header=False) # feat name
        pd.DataFrame(data.indexes['y']).to_csv(os.path.join(dir, name+'_label.name'), index=False, header=False) # label name
        if 'train' in data.indexes: # train/test and X/y splits
            data.get_data('X_train').to_csv(os.path.join(dir, name+'_train.data'), sep=' ', index=False, header=False) # train data
            data.get_data('X_test').to_csv(os.path.join(dir, name+'_test.data'), sep=' ', index=False, header=False) # test data
            data.get_data('y_train').to_csv(os.path.join(dir, name+'_train.solution'), sep=' ', index=False, header=False) # train solution
            data.get_data('y_test').to_csv(os.path.join(dir, name+'_test.solution'), sep=' ', index=False, header=False) # test solution
        else: # only X/y split
            data.get_data('X').to_csv(os.path.join(dir, name+'.data'), sep=' ', index=False, header=False) # data
            data.get_data('y').to_csv(os.path.join(dir, name+'.solution'), sep=' ', index=False, header=False) # solution
    else:
        pd.DataFrame(data.columns).to_csv(os.path.join(dir, name+'_feat.name'), index=False, header=False) # feat name
        if 'train' in data.indexes: # only train/test split
            data.get_data('train').to_csv(os.path.join(dir, name+'_train.data'), sep=' ', index=False, header=False) # train data
            data.get_data('test').to_csv(os.path.join(dir, name+'_test.data'), sep=' ', index=False, header=False) # test data
        else: # no split at all
            data.to_csv(os.path.join(dir, name+'.data'), sep=' ', index=False, header=False) # data
