import tensorflow as tf
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))