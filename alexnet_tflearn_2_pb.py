"""
Load tflearn model into tensflow graph and save to protobuf
Son N. Tran 2016
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('save_path', '/home/funzi/download/alexnet_household.pb', 'Saved file.')
FLAGS = tf.app.flags.FLAGS

def main(_):
    ''' Note that this input is different from standard alexnet where the size is 224x224 '''
    _x = tf.placeholder(tf.float32,[None,227,227,3])
    _y = tf.placeholder(tf.int32,[None,1])


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
    tflearn_model.load(file_path)

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
        ''' Note that this follows the default of tflearn and different from standard alexnet
        where (bias)k=2.0)'''
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
        ''' Note that this follows the default of tflearn and different from standard alexnet
        where (bias)k=2.0)'''
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
        ''' Note that this follows the default of tflearn and different from standard alexnet
        where (bias)k=2.0)'''
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
        fc2 = tf.matmul(self._out_fc1,weights)
        fc2_b = tf.nn.bias_add(fc2,biases)
    
        _out_fc2 = tf.tanh(fc2_b)
        
    with tf.variable_scope('s4_softmax') as scope:
        # NUM_CLASSES = 12
        weights = tf.get_variable('weights',
                                  initializer=tf.constant(tflearn_model.get_weights(s4_fc3.W)))
        biases  = tf.get_variable('biases',
                                  initializer=tf.constant(tflearn_model.get_weights(s4_fc3.b)))
        fc3_out = tf.add(tf.matmul(_out_fc2,weights),biases,name=scope.name)
        
        
        

    session  =  tf.InteractiveSession()
    session.run(tf.initialize_all_variables())
    
    print('Saving alexnet to ',FLAGS.save_path)
    saver = tf.train.Saver(sharded=True)
    model_exporter=exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=_x,ouput_tensor=fc3_out)
    model_exporter.init(sess.graph.as_graph_def(),
                        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(FLAGS.export_version), session)
    print('Done exporting!')

if __name__=='__main__':
    tf.app.run()
