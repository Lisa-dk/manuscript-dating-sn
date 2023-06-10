import os
import sys
import re
import logging

import numpy as np
import tensorflow as tf
from tf_som import SelfOrganizingMap
from sklearn.preprocessing import MinMaxScaler

def get_junctions(file_dir, file):
    """
    Returns all junctions from a file

    :param file_dir: Directory the file to read is in
    :param file:     Name of the .txt file to read
    :return:         List containing all junctions of the file
    """
    junctions = []

    with open(file_dir + file) as f:
        for line in f:
            line = line.rstrip().split(" ")
            junctions.append([float(el) for el in line])

    return junctions


def get_year_junctions(year, files_dir):
    """
    Returns all junctions from a year's files.

    :param year:      The year to obtain the junctions from
    :param files_dir: Directory the files are in
    :return:          List containing all junctions in the given year
    """
    junctions = []
    print(year)
    for file in sorted(os.listdir(files_dir)):
        if file != ".DS_Store" and not os.path.isdir(files_dir + file):
            file_year = re.search("(-?[0-9][0-9][0-9])", file)

            ## Skipping if file belongs to other class than current class
            if int(file_year.group()) < year:
                continue
            elif int(file_year.group()) > year:
                continue
            else:
                #print(file)
                junctions += get_junctions(files_dir, file)
    return junctions


def reduce_features(features):
    """
    Rescales feature vectors between 0 and 1

    :param features: The features to rescale
    :return:         numpy array of rescaled features
    """
    features = np.asarray(features)
    scaler = MinMaxScaler()
    scaler.fit(features)
    data_rescaled = scaler.transform(features)

    return data_rescaled


def train_som(data_size, data, m, n, feat_dim, model_name, weights_init, prev_weights):
    """
    Trains a sub-codebook with SOM.

    :param data_size:    Length of the array
    :param data:         The data to train the SOM with
    :param m:            Number of rows
    :param n:            Number of columns
    :param feat_dim:     Dimensionality of the features
    :param model_name:   Name of the model
    :param weights_init: "PREV" if initialisig with given weigths, otherwise None
    :param prev_weights: Weigths to initialise the SOM with.

    :return: The trained SOM
    """
    batch_size = 128
    learn_start = 0.99
    n_epochs = 500
    n_gpus = 1

    graph = tf.Graph()
    with graph.as_default():
        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        data = tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
        data = data.repeat()

        data = data.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(data)
        next_element = iterator.get_next()
        
        som = SelfOrganizingMap(m, n, feat_dim, max_epochs=n_epochs, batch_size=batch_size, initial_learning_rate=learn_start, 
                            graph=graph, 
                            model_name=model_name, 
                            gpus=n_gpus, 
                            input_tensor=next_element, 
                            input_dataset=data, 
                            session=session,
                            weights_init=weights_init,
                            prev_weights=prev_weights
                            )

        init_op = tf.compat.v1.global_variables_initializer()
        session.run([init_op])
        som.train(num_inputs=data_size)
    return som


def main():
    #device_name = tf.test.gpu_device_name()
    #print(device_name)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    file_dir = './DSS/features/Junclets/'
    ## classes: -470, -365, -198
    year = int(sys.argv[1])                ## year class
    m = n = int(sys.argv[2])               ## Subcodebook size
    prev_weights_file = sys.argv[3]        ## weights file to use for further training. If no prev -> "None"

    weights_file_dir = './DSS/500_weights/' + str(year) + '/'

    if not os.path.exists(weights_file_dir):
        os.makedirs(weights_file_dir)

    ## Get junctions for given year
    junctions_year = get_year_junctions(year, file_dir)
    print(len(junctions_year), 'size:', m, 'year: ', year)
    data = np.array(junctions_year)

    data_size = len(data)
    model_name = 'som_' + str(year) + '_' + str(m)
    weights_file = weights_file_dir + 'size_' + str(m)

    ## Loading previous weights if relevant
    if prev_weights_file != 'None':
        prev_weights_dir = './DSS/500_weights/' 
        restore_weights = prev_weights_file
        prev_weights = np.load(prev_weights_dir + restore_weights)
        weights_init = "PREV"
    else:
        prev_weights = None
        weights_init = None
    
    # Training the SOM and saving the weights
    som = train_som(data_size, data, m, n, 120, model_name, weights_init, prev_weights)
    np.save(weights_file + '.npy', som.output_weights)
        

if __name__ == "__main__":
    main()