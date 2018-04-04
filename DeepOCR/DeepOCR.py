import numpy as np
import argparse
import os
import string
# import sys
from skimage.io import imread
from sklearn.model_selection import ShuffleSplit
from TFANN import ANNC
import tensorflow as tf
from scipy.stats import mode as Mode

NB_CHAR = len(string.ascii_letters + string.digits + ' ' + string.punctuation)    # Number of possible characters
MAX_CHAR = 64           # Max # characters per block
IM_SIZE = [21, 1088, 3]       # Image size for CNN
NB_NET = 5                # Use 5 networks with a bagging approach
# Shared placeholders for images and target data
TENSOR_IMAGE = tf.placeholder("float", [None] + IM_SIZE, name='TENSOR_IMAGE') 
TENSOR_STRING = tf.placeholder("float", [None, MAX_CHAR, NB_CHAR], name='TENSOR_STRING')


def divide_into_sub_images(image):
    """
    input: image
    Divides an image into chunks to feed into OCR net
    outuput:
    """
    im_height, im_width, im_depth = image.shape
    H= im_height // IM_SIZE[0]
    W = im_width // IM_SIZE[1]
    HP = IM_SIZE[0] * H
    WP = IM_SIZE[1] * W
    image = image[0:HP, 0:WP]     # Discard any extra pixels

    return image.reshape(H, IM_SIZE[0], -1, IM_SIZE[1], im_depth).swapaxes(1, 2).reshape(-1, IM_SIZE[0], IM_SIZE[1], im_depth)


def make_network(net_name ='ocrnet'):
    # Architecture of the neural network
    # The input volume is reduce to the shape of the output in conv layers
    # 18 / 2 * 3 * 3 = 1 and 640 / 2 * 5 = 64 output.shape
    ws = [('C', [5, 5,  3, NB_CHAR // 2], [1, 2, 2, 1]), ('AF', 'relu'),
          ('C', [4, 4, NB_CHAR // 2, NB_CHAR], [1, 3, 1, 1]), ('AF', 'relu'),
          ('C', [3, 5, NB_CHAR,      NB_CHAR], [1, 3, 5, 1]), ('AF', 'relu'),
          ('R', [-1, MAX_CHAR, NB_CHAR])]
    # Create the neural network in TensorFlow
    return ANNC(IM_SIZE, ws, batchSize=512, learnRate=2e-5, maxIter=64, name=net_name, reg=1e-5, tol=1e-2,
                verbose=True, X=TENSOR_IMAGE, Y=TENSOR_STRING)


def fit_model(cnnc, A, Y, T, FN):
    print('Fitting model...')
    ss = ShuffleSplit(n_splits = 1)
    trn, tst = next(ss.split(A))
    # Fit the network
    cnnc.fit(A[trn], Y[trn])
    # The predictions as sequences of character indices
    YH = []
    for i in np.array_split(np.arange(A.shape[0]), 32): 
        YH.append(cnnc.predict(A[i]))
    YH = np.vstack(YH)
    # Convert from sequence of char indices to strings
    PS = np.array([''.join(YHi) for YHi in YH])
    # Compute the accuracy
    S1 = SAcc(PS[trn], T[trn])
    S2 = SAcc(PS[tst], T[tst])
    print('Train: ' + str(S1))
    print('Test: ' + str(S2))
    for PSi, Ti, FNi in zip(PS, T, FN):
        if np.random.rand() > 0.99: # Randomly select rows to print
            print(FNi + ': ' + Ti + ' -> ' + PSi)
    print('Fitting with CV data...')
    # Fit remainder
    cnnc.SetMaxIter(4)
    cnnc.fit(A, Y)
    return cnnc

    
def fuse_results(tensor):  # Take mode result from 5 networks
    return (Mode(np.stack([tensor_i.ravel() for tensor_i in tensor]))[0]).reshape(tensor[0].shape)


def image_to_string(image):
    """
    Uses OCR to transform an image into a string
    """
    sub_image = divide_into_sub_images(image)
    sub_image_shape = sub_image_shape(image)
    YH = fuse_results(TFS.run(YHL, feed_dict={TENSOR_IMAGE: sub_image}))
    return join_string(CNN[-1]._classes[YH], sub_image_shape)


def join_string(YH, ss):
    """
    Rejoin substrings according to position of subimages
    """
    YH = np.array([''.join(YHi) for YHi in YH]).reshape(ss)
    return '\n'.join(''.join(YHij for YHij in YHi) for YHi in YH)


def load_data(file_path='../'):
    """
    Loads the OCR dataset. A is matrix of images (NIMG, Height, Width, Channel).
    Y is matrix of characters (NIMG, MAX_CHAR)
    file_path:     Path to OCR data folder
    return: Data Matrix, Target Matrix, Target Strings
    """
    training_csv_file_path = os.path.join(file_path, 'Training.csv')
    A, Y, T, FN = [], [], [], []
    with open(training_csv_file_path) as F:
        for i, Li in enumerate(F):
            im_filename_i, string_i = Li.strip().split(',')                     # filename,string
            T.append(string_i)
            A.append(imread(im_filename_i)[:, :, :3])   # Read image and discard alpha channel
            Y.append(list(string_i) + [' '] * (MAX_CHAR - len(string_i)))   # Pad strings with spaces
            FN.append(im_filename_i)
    return np.stack(A), np.stack(Y), np.stack(T), np.stack(FN)


def SAcc(T, PS):
    return sum(sum(i == j for i, j in zip(S1, S2)) / len(S1) for S1, S2 in zip(T, PS)) / len(T)


def sub_image_shape(image):
    """
    Get number of (rows, columns) of subimages
    """
    h, w, c = image.shape
    return h // IM_SIZE[0], w // IM_SIZE[1]


def load_nets():
    CNN = [make_network('ocrnet' + str(i)) for i in range(NB_NET)]
    if not CNN[-1].RestoreModel('TFModel/', 'ocrgraph'):
        A, Y, T, FN = load_data()
        for CNNi in CNN:
            fit_model(CNNi, A, Y, T, FN)
        CNN[-1].SaveModel(os.path.join('TFModel', 'ocrgraph'))
        with open('TFModel/_classes.txt', 'w') as F:
            F.write('\n'.join(CNN[-1]._classes))
    else:
        with open('TFModel/_classes.txt') as F:
            cl = F.read().splitlines()
        for CNNi in CNN:
            CNNi.RestoreClasses(cl)
    return CNN


if __name__ == "__main__":
    P = argparse.ArgumentParser(description='Deep learning based OCR')
    P.add_argument('-f', action='store_true', help='Force model training')
    P.add_argument('Img', metavar='image', type=str, nargs='+', help='Image files')
    PA = P.parse_args()
    if PA.f:
        CNN = load_nets()  # CNNs
        YHL = [CNNi.YHL for CNNi in CNN]  # Prediction placeholders
        TFS = CNN[-1].GetSes()  # Get tensorflow session
    for img in PA.Img:
        image = imread(img)[:, :, :3]    # Read image and discard alpha
        result_string = image_to_string(image)
        print(result_string)


# from PIL import Image
# im = Image.open('/Users/rpgb/GitHubRepos/pythonml/releves/Test_tif.tif')
# print(im.size)
# im.show()

# #  to read images from pdf
# from pdf2image import convert_from_path
# image2 = convert_from_path('/Users/rpgb/GitHubRepos/pythonml/releves/Test_pdf.pdf')
# image2[0].show()
