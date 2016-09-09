"""
Load tflearn model into tensflow graph and save to protobuf
Son N. Tran 2016
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import googletest
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile

import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import os

tf.app.flags.DEFINE_string('ckpt_prefix', '/home/tra161/WORK/projects/object_reg/experiments/household_tflearn_model/saved_checkpoint', 'checkpoint prefix.')
tf.app.flags.DEFINE_string('temp_dir', '/home/tra161/WORK/projects/object_reg/experiments/household_tflearn_model/', 'checkpoint prefix.')
tf.app.flags.DEFINE_string('ckpt_state_name', 'checkpoint_state', 'Checkpoint state.')
tf.app.flags.DEFINE_string('inp_graph_name', 'input_graph.pb', 'Input graph.')
tf.app.flags.DEFINE_string('outp_graph_name', 'output_graph.pb', 'Output graph.')

tf.app.flags.DEFINE_string('model_path', '/home/tra161/WORK/projects/object_reg/experiments/household_tflearn_model/model_alexnet-60000', 'Tflearn model file.')

FLAGS = tf.app.flags.FLAGS

def main(_):

    # LOAD  TFLEARN MODEL
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
    tflearn_model.load(FLAGS.model_path)
    
    with tf.Graph().as_default():
        ### Note that this input is different from standard alexnet where the size is 224x224 
        _x = tf.placeholder(tf.float32,[None,227,227,3])
        _y = tf.placeholder(tf.int32,[None,1])

        #print(tflearn_model.get_weights(s4_fc3.b))
        
        # Create tensorflow graph
     
        ####################################################################################
        #Stage 1: 1 conv + 1 pooling + 1 norm 
        ####################################################################################
        with tf.variable_scope('s1_conv1') as scope: # Stage 1 convolution 1
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s1_conv1.W)))
            conv   = tf.nn.conv2d(_x,kernel,strides=[1,4,4,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s1_conv1.b)))
            conv_b = tf.nn.bias_add(conv,biases)
         
            out_s1_conv1 = tf.nn.relu(conv_b,name=scope.name)
         
            #Pooling at stage 1
            _pool1 = tf.nn.max_pool(out_s1_conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                    padding='SAME',name='pool1')
            #Local response norm at stage 1
            ### Note that this follows the default of tflearn and different from standard alexnet
            ### where (bias)k=2.0)
            _norm1 = tf.nn.lrn(_pool1,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm1')
         
            ####################################################################################
            # Stage 2: 1 conv + 1 pool + 1 norm
            ####################################################################################
        with tf.variable_scope('s2_conv1'):
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s2_conv1.W)))
            conv   = tf.nn.conv2d(_norm1,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s2_conv1.b)))
            conv_b = tf.nn.bias_add(conv,biases)
            
            _out_s2_conv1 = tf.nn.relu(conv_b,name=scope.name)
            
            #Pooling at stage 2
            _pool2 = tf.nn.max_pool(_out_s2_conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                    padding='SAME',name='pool2')
            #Local response norm at stage 1
            ### Note that this follows the default of tflearn and different from standard alexnet
            ### where (bias)k=2.0)
            _norm2 = tf.nn.lrn(_pool2,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm2')
            
        ####################################################################################
        #Stage 3 : 3 conv + 1 pooling + 1 norm
        ####################################################################################
        with tf.variable_scope('s3_conv1') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv1.W)))
            conv   = tf.nn.conv2d(_pool2,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv1.b)))
            conv_b = tf.nn.bias_add(conv,biases)
            
            _out_s3_conv1 = tf.nn.relu(conv_b,name=scope.name)
        
        with tf.variable_scope('s3_conv2') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv2.W)))
            conv   = tf.nn.conv2d(_out_s3_conv1,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv2.b)))
            conv_b = tf.nn.bias_add(conv,biases)
        
            _out_s3_conv2 = tf.nn.relu(conv_b,name=scope.name)

        with tf.variable_scope('s3_conv3') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv3.W)))
            conv   = tf.nn.conv2d(_out_s3_conv2,kernel,[1,1,1,1],padding='SAME')
            biases = tf.get_variable('biases',
                                     initializer=tf.constant(tflearn_model.get_weights(s3_conv3.b)))
            conv_b = tf.nn.bias_add(conv,biases)
        
            _out_s3_conv3 = tf.nn.relu(conv_b,name=scope.name)
        
            #Pooling at stage 3
            _pool3 = tf.nn.max_pool(_out_s3_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],
                                    padding='SAME',name='pool3')
            #Local response norm at stage 1
            ### Note that this follows the default of tflearn and different from standard alexnet
            ### where (bias)k=2.0)
            _norm3 = tf.nn.lrn(_pool3,5,bias=1.0,alpha=0.0001,beta=0.75,name='norm2')
        ####################################################################################
        #Stage 4 : 2 fully-connected layers + 1 softmax layer
        ####################################################################################
        with tf.variable_scope('s4_fc1') as scope:
            norm3_node_num   = int(np.prod(_norm3.get_shape()[1:]))
            norm3_vectorized = tf.reshape(_norm3,[-1,norm3_node_num])
            weights = tf.get_variable('weights',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc1.W)))
            biases  = tf.get_variable('biases',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc1.b)))
            fc1 = tf.matmul(norm3_vectorized,weights)
            fc1_b = tf.nn.bias_add(fc1,biases)
            
            _out_fc1 = tf.tanh(fc1_b)
            
        with tf.variable_scope('s4_fc2') as scope:
            weights = tf.get_variable('weights',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc2.W)))
            biases  = tf.get_variable('biases',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc2.b)))
            fc2 = tf.matmul(_out_fc1,weights)
            fc2_b = tf.nn.bias_add(fc2,biases)
            
            _out_fc2 = tf.tanh(fc2_b)
            
        with tf.variable_scope('s4_softmax') as scope:
            # NUM_CLASSES = 12
            weights = tf.get_variable('weights',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc3.W)))
            biases  = tf.get_variable('biases',
                                      initializer=tf.constant(tflearn_model.get_weights(s4_fc3.b)))
            fc3_out = tf.add(tf.matmul(_out_fc2,weights),biases,name='final_tensor')
        
        
        session  =  tf.Session()
        session.run(tf.initialize_all_variables())

        ## Method 1
        #saver = tf.train.Saver()
        #saver.save(session,FLAGS.ckpt_prefix,global_step=0, latest_filename=FLAGS.ckpt_state_name)
        #tf.train.write_graph(session.graph.as_graph_def(),FLAGS.temp_dir,FLAGS.inp_graph_name)
        ### Method 2
        output_graph_def = graph_util.convert_variables_to_constants(session,session.graph.as_graph_def(),['s4_softmax/final_tensor'])
        output_graph_path = os.path.join(FLAGS.temp_dir, FLAGS.outp_graph_name)
        with gfile.FastGFile(output_graph_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

                            
    #### SAVE TO FILE ###
    print('Saving alexnet to ',FLAGS.temp_dir)
    # We save out the graph to disk, and then call the const conversion
    # routine.
    #input_graph_path = os.path.join(FLAGS.temp_dir, FLAGS.inp_graph_name)
    #input_saver_def_path = ""
    #input_binary = False
    #input_checkpoint_path = FLAGS.ckpt_prefix + "-0"
    #output_node_names = "output_node"
    #restore_op_name = "save/restore_all"
    #filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(FLAGS.temp_dir, FLAGS.outp_graph_name)
    #clear_devices = False

    #freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
    #                          input_binary, input_checkpoint_path,
    #                          output_node_names, restore_op_name,
    #                          filename_tensor_name, output_graph_path,
    #                          clear_devices,"")

    ''' LOAD PB file. code from: https://github.com/tensorflow/tensorflow/blob/00440e99ffb1ed1cfe4b4ea650e0c560838a6edc/tensorflow/python/tools/freeze_graph_test.py#L68'''
    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        self.assertEqual(4, len(output_graph_def.node))
        for node in output_graph_def.node:
            self.assertNotEqual("Variable", node.op)

        with tf.Session() as sess:
            output_node = sess.graph.get_tensor_by_name("output_node:0")
            output = sess.run(output_node)
            self.assertNear(2.0, output, 0.00001)
    '''
if __name__=='__main__':
    tf.app.run()
