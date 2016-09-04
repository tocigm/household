import tensorflow as tf



with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model_alexnet-58.meta')
    new_saver.restore(sess, 'model_alexnet-58')

