# Function to read and write AutoML format
# TODO

def read_automl(input_dir, basename):
    """ Read files in AutoML format.
        TODO
    """
    pass
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

def to_automl():
    """ Write files in AutoML format.
        TODO
    """
    pass
