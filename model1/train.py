import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='x')
w = tf.Variable(tf.truncated_normal(shape=[1, 2]), name='w')
y = tf.add(x, w, name="y")

init = tf.initialize_all_variables()
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init)
    print("w", sess.run(w))
    print("y", sess.run(y, feed_dict={x: [[6, 7]]}))
    save_path = saver.save(sess, "./model.ckpt")