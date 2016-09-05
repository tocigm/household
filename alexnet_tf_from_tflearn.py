"""
Alexnet
Son N. Tran
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class AlexnetGraph(object):

    def __init__(self,is_training,conf):
        ''' Note that this input is different from standard alexnet where the size is 224x224 '''
        self._x = tf.placeholder(tf.float32,[None,227,227,3])
        self._y = tf.placeholder(tf.int32,[None,1])

        if conf.load:
            self.load_graph_from_tflearn(conf.load_file)
            
    def load_graph_from_tflearn(self,file_path):
        inp = input_data(shape=[None, 227, 227, 3])
        s1_conv1 = conv_2d(inp, 96, 11, strides=4, activation='relu')
        pool1 = max_pool_2d(s1_conv1, 3, strides=2)
        norm1 = local_response_normalization(pool1)
    
        s2_conv1 = conv_2d(norm1, 256, 5, activation='relu')
        pool2 = max_pool_2d(s2_conv1, 3, strides=2)
        norm2 = local_response_normalization(pool2)
    
        s3_conv1 = conv_2d(norm2, 384, 3, activation='relu')
        s3_conv2 = conv_2d(s3_conv1, 384, 3, activation='relu')
        s3_conv3 = conv_2d(s3_conv2, 256, 3, activation='relu')
        pool3 = max_pool_2d(s3_conv3, 3, strides=2)
        norm3 = local_response_normalization(pool3)
        
        s4_fc1 = fully_connected(norm3, 4096, activation='tanh')
        s4_fc1_do = dropout(s4_fc1, 0.5)
        s4_fc2 = fully_connected(s4_fc1_do, 4096, activation='tanh')
        s4_fc2_do = dropout(s4_fc2, 0.5)
        s4_fc3 = fully_connected(s4_fc2_do, 12, activation='softmax')
        network = regression(s4_fc3, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

        tflearn_model = tflearn.DNN(network)
        tflearn_model.load(file_path)

        #print(tflearn_model.get_weights(s4_fc3.b))
        
    
        ####################################################################################
        #Stage 1: 1 conv + 1 pooling + 1 norm 
        ####################################################################################
        with tf.variable_scope('s1_conv1') as scope: # Stage 1 convolution 1
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s1_conv1.W)))
            conv   = tf.nn.conv2d(self._x,kernel,strides=[1,4,4,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s1_conv1.b)))
            conv_b = tf.nn.bias_add(conv,biases)

            out_s1_conv1 = tf.nn.relu(conv_b,name=scope.name)

            #Pooling at stage 1
            self._pool1 = tf.nn.max_pool(out_s1_conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                   padding='SAME',name='pool1')
            #Local response norm at stage 1
            ''' Note that this follows the default of tflearn and different from standard alexnet
            where (bias)k=2.0)'''
            self._norm1 = tf.nn.lrn(self._pool1,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm1')
             
         ####################################################################################
         # Stage 2: 1 conv + 1 pool + 1 norm
         ####################################################################################
        with tf.variable_scope('s2_conv1'):
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s2_conv1.W)))
            conv   = tf.nn.conv2d(self._norm1,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s2_conv1.b)))
            conv_b = tf.nn.bias_add(conv,biases)

            self._out_s2_conv1 = tf.nn.relu(conv_b,name=scope.name)

            #Pooling at stage 2
            self._pool2 = tf.nn.max_pool(self._out_s2_conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                   padding='SAME',name='pool2')
            #Local response norm at stage 1
            ''' Note that this follows the default of tflearn and different from standard alexnet
            where (bias)k=2.0)'''
            self._norm2 = tf.nn.lrn(self._pool2,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm2')
             
        ####################################################################################
        #Stage 3 : 3 conv + 1 pooling + 1 norm
        ####################################################################################
        with tf.variable_scope('s3_conv1') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv1.W)))
            conv   = tf.nn.conv2d(self._pool2,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv1.b)))
            conv_b = tf.nn.bias_add(conv,biases)

            self._out_s3_conv1 = tf.nn.relu(conv_b,name=scope.name)

        with tf.variable_scope('s3_conv2') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv2.W)))
            conv   = tf.nn.conv2d(self._out_s3_conv1,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv2.b)))
            conv_b = tf.nn.bias_add(conv,biases)

            self._out_s3_conv2 = tf.nn.relu(conv_b,name=scope.name)

        with tf.variable_scope('s3_conv3') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv3.W)))
            conv   = tf.nn.conv2d(self._out_s3_conv2,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv3.b)))
            conv_b = tf.nn.bias_add(conv,biases)

            self._out_s3_conv3 = tf.nn.relu(conv_b,name=scope.name)

            #Pooling at stage 3
            self._pool3 = tf.nn.max_pool(self._out_s3_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],
                                        padding='SAME',name='pool3')
            #Local response norm at stage 1
            ''' Note that this follows the default of tflearn and different from standard alexnet
            where (bias)k=2.0)'''
            self._norm3 = tf.nn.lrn(self._pool3,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm2')
        ####################################################################################
        #Stage 4 : 2 fully-connected layers + 1 softmax layer
        ####################################################################################
        with tf.variable_scope('s4_fc1') as scope:
            norm3_node_num   = int(np.prod(self._norm3.get_shape()[1:]))
            print(norm3_node_num)
            norm3_vectorized = tf.reshape(self._norm3,[-1,norm3_node_num])
            weights = tf.get_variable('weights',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc1.W)))
            biases  = tf.get_variable('biases',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc1.b)))
            fc1 = tf.matmul(norm3_vectorized,weights)
            fc1_b = tf.nn.bias_add(fc1,biases)
            
            self._out_fc1 = tf.tanh(fc1_b)
            
        with tf.variable_scope('s4_fc2') as scope:
            weights = tf.get_variable('weights',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc2.W)))
            biases  = tf.get_variable('biases',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc2.b)))
            fc2 = tf.matmul(self._out_fc1,weights)
            fc2_b = tf.nn.bias_add(fc2,biases)
            
            self._out_fc2 = tf.tanh(fc2_b)
            
        with tf.variable_scope('s4_softmax') as scope:
            # NUM_CLASSES = 12
            weights = tf.get_variable('weights',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc3.W)))
            biases  = tf.get_variable('biases',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc3.b)))
            fc3_out = tf.add(tf.matmul(self._out_fc2,weights),biases,name=scope.name)

            
            # Prediction accuracy
            self._pred = pred = tf.argmax(fc3_out,1)

            '''
            #Cost: cross entropy
            cross_en =  tf.nn.sparse_softmax_cross_entropy_with_logits(
                fc3_out,tf.cast(self._y,tf.int64),name='cross_entropy_per_example')
            cross_en_mean =
            self._cost = cost =
            if not is_training:
                return
            '''
            
            ''' TODO
            self._lr = tf.Variable(0.0,trainable=False)
            tvars = tf.trainable_variables()
            grads,_ = tf.gradients(cost,vars)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(zip(grads,tvars))
            
            '''
            
class Alexnet(object):

    def __init__(self,config,dataset):
        self._dataset = dataset
        self._config  = config
    # Predict
    def predict(self,X,Y=None):
        acc = -1
        with tf.Graph().as_default():
            mpredict  = AlexnetGraph(False,self._config)
            
            with tf.Session() as session:
                
                init = tf.initialize_all_variables()
                session.run(init)
                
                pred = session.run([mpredict._pred],{mpredict._x:X})
                acc = np.mean(np.equal(pred,Y))
        return acc, pred
    
                                    
class TrainConfig():
    load = False

class PredConfig():
    load = True
    load_file = '/home/tra161/WORK/projects/object_reg/experiments/household_tflearn_model/model_alexnet-60000'

def test_load_tflearn_model():
    load_tflearn_model('/home/tra161/WORK/projects/object_reg/experiments/household_tflearn_model/model_alexnet-60000',m=None)
    
def test_training():
    cifar_data = CIFAR10()
    
def test_tflearn_load():
    #household_data = Household()
    #X,Y = household_data.evaluate_data()
    X = np.random.rand(100,227,227,3)
    Y = np.random.randint(12,size=100)
    print(Y.shape)
    alexnet = Alexnet(PredConfig(),dataset=None)
    acc,_ = alexnet.predict(X,Y)
    print(acc)

if __name__== "__main__":
    test_tflearn_load()
