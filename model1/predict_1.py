import tensorflow as tf


# 加载模型需要重新定义模型参数
# w = tf.Variable(tf.truncated_normal(shape=[1, 2]), name='w')
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, './model.ckpt')
#     print(sess.run(w))

saver = tf.train.import_meta_graph("./model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "./model.ckpt")
    x = tf.get_default_graph().get_operation_by_name('x').outputs[0]
    y = tf.get_default_graph().get_operation_by_name('y').outputs[0]
    w = tf.get_default_graph().get_tensor_by_name('w:0')

    print('w', sess.run(w))
    print(sess.run(y, feed_dict={x: [[5, 6]]}))














