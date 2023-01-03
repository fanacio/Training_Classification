import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
from PIL import Image
import numpy as np

MNIST_data_folder = 'MNIST_data_folder'
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=False) #MNIST_data_folder是数据集的目录
#imgs, labels = mnist.test.images, mnist.test.labels #生成测试集图片
#imgs, labels = mnist.validation.images, mnist.validation.labels  #生成验证集图片
imgs, labels = mnist.train.images, mnist.train.labels #生成训练集图片
for i in range(10):
    if not os.path.exists(str(i)):
        os.makedirs(str(i))
cnt = [0 for i in range(10)]
for i in range(imgs.shape[0]):
    array = (imgs[i].reshape((28, 28)) * 255).astype(np.uint8)
    cnt[labels[i]] += 1
    img = Image.fromarray(array, 'L')
    img.save(str(labels[i]) + '/' + str(cnt[labels[i]]) + '.jpg')

