import tensorflow as tf
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4],minval=-10,maxval=10))
# dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
#
# print(dataset1)
# print(dataset1.output_shapes)
# print(dataset1.output_types)
#
# print(dataset2)
# print(dataset2.output_shapes)
# print(dataset2.output_types)
#
# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset3.output_shapes)
# print(dataset3.output_types)
#
# dataset4 = tf.data.Dataset.from_tensor_slices(
#     {'a':tf.random_uniform([4]),
#      'b':tf.random_uniform([4,10],minval=-10,maxval=10,dtype=tf.float32)})
# print(dataset4.output_shapes)
# print(dataset4.output_types)
#
# with tf.Session() as sess:
#     print(sess.run(tf.random_uniform([4])))
#
# dataset5 = tf.data.Dataset.range(100)
# iterator = dataset5.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(10):
#         value = sess.run(next_element)
#         print(value)

# max_value = tf.placeholder(tf.int64, shape=[])
# dataset = tf.data.Dataset.range(max_value)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={max_value:15})
#     for i in range(12):
#         value = sess.run(next_element)
#         assert i == value
#         print(value)
#
#     sess.run(iterator.initializer, feed_dict={max_value:100})
#
#     for i in range(12):
#         value = sess.run(next_element)
#         print(value)

# training_dataset = tf.data.Dataset.range(10).map(
#     lambda x:x + tf.random_uniform([],-10,10,tf.int64))
# validation_dataset = tf.data.Dataset.range(50)
# iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
#                                            training_dataset.output_shapes)
# next_element = iterator.get_next()
#
# training_init_op = iterator.make_initializer(training_dataset)
# validation_init_op = iterator.make_initializer(validation_dataset)
# with tf.Session() as sess:
#     for _ in range(1):
#         sess.run(training_init_op)
#         for _ in range(6):
#             print(sess.run(next_element))
#
#         sess.run(validation_init_op)
#         for _ in range(10):
#             print(sess.run(next_element))

# training_dataset = tf.data.Dataset.range(50).map(
#     lambda x:x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
# validation_dataset = tf.data.Dataset.range(50)
#
# handle = tf.placeholder(tf.string, shape=[])
# iterator = tf.data.Iterator.from_string_handle(
#     handle, training_dataset.output_types, training_dataset.output_shapes)
# next_element = iterator.get_next()
#
# training_iterator = training_dataset.make_one_shot_iterator()
# validation_iterator = validation_dataset.make_initializable_iterator()
# with tf.Session() as sess:
#
#     training_handle = sess.run(training_iterator.string_handle())
#     validation_handle = sess.run(validation_iterator.string_handle())
#     i = 0
#     while i < 2:
#         i += 1
#         for _ in range(20):
#             print(sess.run(next_element, feed_dict={handle:training_handle}))
#
#         sess.run(validation_iterator.initializer)
#
#         for _ in range(10):
#             print(sess.run(next_element, feed_dict={handle:validation_handle}))

# dataset = tf.data.Dataset.range(5)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# result = tf.add(next_element, next_element)
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     print(sess.run(result))
#     print(sess.run(result))
#     print(sess.run(result))
#     print(sess.run(result))
#     print(sess.run(result))
#
#     try:
#         sess.run(result)
#     except tf.errors.OutOfRangeError:
#         print('end of dataset')
#
#
#     sess.run(iterator.initializer)
#     while True:
#         try:
#             print(sess.run(result))
#         except:
#             print('End of Dataset')
#             break

# saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
#
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#
#     if should_checkpoint:
#         saver.save(path_to_checkpoint)
#
#
# with tf.Session() as sess:
#     saver.restore(sess, path_to_checkpoint)

# Batching dataset elements
#
# inc_dataset = tf.data.Dataset.range(100)
# dec_dataset = tf.data.Dataset.range(0,100000,100)
# dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
# batched_dataset = dataset.batch(4)
#
# iterator = batched_dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
# with tf.Session() as sess:
#     print(sess.run(next_element))
#     print(sess.run(next_element))
#     print(sess.run(next_element))


dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                                   #      [5, 5, 5, 5, 5, 0, 0],
                                   #      [6, 6, 6, 6, 6, 6, 0],
                                   #      [7, 7, 7, 7, 7, 7, 7]]