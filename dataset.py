import os
import tensorflow as tf
from PIL import Image
import numpy as np


class Dataset:

    def __init__(self, path):
        self.path = path
        # 定义位置

        self.filenames = os.listdir(path)
        # print(self.filenames)
        # 使用os.listdir()输出路径下文件名的列表
        self.labels = list(map(lambda filename: int(filename.split('.')[0]), self.filenames))
        # 使用文件名作为标签
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        # 生成数据集
        self.dataset = self.dataset.map(lambda filename, label: tuple(tf.py_func(
            self._read_py_function, [filename, label], [tf.float32, label.dtype])))
        # 使用py_func生成数据集
        self.dataset = self.dataset.shuffle(buffer_size=3)
        # 将数据集随机打乱
        self.dataset = self.dataset.repeat()
        # 设置数据集可以重复
        self.dataset = self.dataset.batch(2)
        # 设计取2张图片为一批
        iterator = self.dataset.make_one_shot_iterator()
        # 生成one_hot迭代器
        self.next_element = iterator.get_next()
        # 生成get_next_element的作用

    def get_batch(self, sess):
        return sess.run(self.next_element)

    def _read_py_function(self, filename, label):
        _filename = bytes.decode(filename)
        image_path = os.path.join(self.path, _filename)
        im = Image.open(image_path)
        im = im.resize((300, 300))
        image_data = np.array(im, dtype=np.float32) / 255. - 0.5

        return image_data, label


if __name__ == '__main__':
    myDataset = Dataset('./data')

    with tf.Session() as sess:
        # 随机取用
        xs, ys = myDataset.get_batch(sess)
        print(xs, ys)

        xs, ys = myDataset.get_batch(sess)
        print(xs, ys)


