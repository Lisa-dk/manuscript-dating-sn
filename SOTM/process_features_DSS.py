import sys
import os

import numpy as np
from tqdm import tqdm
from numpy import linalg, subtract


def process_junclets2(file_dir, codebook, save_dir):
    """
    Computes feature vectors for a grapheme-based feature by mapping 
    the graphemes to a codebook and obtaining a normalised histogram.

    :param file_dir: Directory to read the files from
    :param codebook: The trained codebook
    :param save_dir: Directory to store the feature files in
    """

    # Iterates over all manuscripts
    for file in tqdm(sorted(os.listdir(file_dir))):
        if file != ".DS_Store" and not os.path.isdir(file_dir + file):
            counts = [0 for _ in range(len(codebook))]
            with open(file_dir + file) as f:
                # Iterating over all graphemes of a .txt file and obtaining the histogram
                for line in f:
                    line = line.rstrip().split(" ")
                    junclet = np.array([float(el) for el in line])
                    # Finding grapheme in codebook most similar to current one
                    distances = linalg.norm(subtract(junclet, codebook), axis=-1) # Euclidean distance
                    counts[np.argmin(distances)] += 1
                # Getting the normalised histogram
                sum_feat = np.sum(counts)
                features_file = [el / sum_feat for el in counts]

            # Storing the feature vector
            with open(save_dir + file, 'w+') as feat_f:
                for feat in features_file:
                    feat_f.write(str(feat) + ' ') 


def main():
    size = int(sys.argv[1])                             # sub-codebook size to compute the features for
    weights_file_name = 'size_' + sys.argv[1] + '.npy'  # Standard name of files containing weights of sub-codebook
    save_directory = './DSS/features/Junclets_/size_' + sys.argv[1] + '/'  # Directory to save features in
    file_dir = './DSS/features/Junclets/'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    start_year = -470             # Earliest key year/class
    n_classes = 3                 # Number of classes
    h = size * n_classes          # Size of the codebook
    classes = [-470, -365, -198]  # List of key years/classes

    codebook = []

    # Obtaining the codebook by concatenating sub-codebooks for all classes
    for year in tqdm(classes):
        dir_name = './DSS/500_weights/' + str(year) + '/'
        if year == start_year:
            weights = np.load(dir_name + weights_file_name)
            codebook = weights.tolist()
        else:
            weights = np.load(dir_name + weights_file_name)
            codebook += weights.tolist()
    
    print(len(codebook), len(codebook[0]))
    process_junclets2(file_dir, np.array(codebook), save_directory)


if __name__ == "__main__":
    main()