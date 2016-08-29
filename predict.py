from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
from PIL import Image

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# Load path/class_id image file:
dataset_file = 'train.txt'

# Build a HDF5 dataset (only required once)

#from tflearn.data_utils import build_hdf5_image_dataset
#build_hdf5_image_dataset(dataset_file, image_shape=(224, 224), mode='file', output_path='dataset.h5', categorical_labels=True, normalize=True)

#from tflearn.data_utils import image_preloader
#X, Y = image_preloader(dataset_file, image_shape=(224, 224),   mode='file', categorical_labels=True,   normalize=True)

# Load HDF5 dataset
#import h5py
#h5f = h5py.File('dataset.h5', 'r')
#X = h5f['X']
#Y = h5f['Y']


def load_image(img_path):
    img = Image.open(img_path)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def load_data(datafile, num_clss, save=False, save_path='dataset.pkl'):
    train_list = open(datafile,'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        print(fpath)
        img = load_image(fpath)
        img = resize_image(img,224,224)
        np_img = pil_to_nparray(img)
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(num_clss)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_from_pkl(dataset_file):
    X, Y = pickle.load(open(dataset_file, 'rb'))
    return X,Y


def create_vgg16(num_classes):
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')

    network = regression(network, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return network


def train(network, X, Y):
    # Training
    model = tflearn.DNN(network, checkpoint_path='vgg16_household',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output')
    model.fit(X, Y, n_epoch=100000, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=2, snapshot_step=5000,
              snapshot_epoch=False, run_id='vgg_household')

def predict(network, modelfile,images):
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)

if __name__ == '__main__':
    #X, Y = load_data('train.txt', 42)
    #X, Y = load_from_pkl('test.pkl')
    net = create_vgg16(41)
    model = tflearn.DNN(net)
    model.load("vgg16_household-80000")

    testfile = open("list_test.txt","r")
    resultfile = open("result.txt", "w")

    object_name = []

    #read object name
    objectfile = open("cate.txt", "r")
    objectlist = objectfile.readlines()
    for obj in objectlist:
        object_name.append(obj.replace("\n", ""))

    print(len(object_name))

    lines = testfile.readlines()
    for index, line in enumerate(lines):
        line = line.replace("\n","")
        img = load_image(line)
        img = resize_image(img, 224, 224)
        img = pil_to_nparray(img)
        pred = model.predict([img])
        pred = np.array(pred[0])
        object_index = pred.argsort()[-5:][::-1]
        resultfile.write(line)
        for idx in object_index:
            prob = "%.3f" % pred[idx]
            resultfile.write("\t" + object_name[idx] + ":" + prob)
        resultfile.write("\n")




