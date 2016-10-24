# pylint:disable=C0103
########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
import operator

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
try:
    from functools import reduce
except:
    pass

from caffe_classes import class_names

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 1e1
REG_WEIGHT = 0e0
TV_WEIGHT = 1e0
LEARNING_RATE = 1e1
ITERATIONS = 500


def preprocess(image):
    return image - [123.68, 116.779, 103.939]


def unprocess(image):
    image = image + [123.68, 116.779, 103.939]
    return image


def build_net(input):
    layers = []
    layers.extend(['conv1_1', 'conv1_2', 'pool1'])
    layers.extend(['conv2_1', 'conv2_2', 'pool2'])
    layers.extend(['conv3_1', 'conv3_2', 'conv3_3', 'pool3'])
    layers.extend(['conv4_1', 'conv4_2', 'conv4_3', 'pool4'])
    layers.extend(['conv5_1', 'conv5_2', 'conv5_3', 'pool5'])
    # layers.extend(['fc6', 'fc7', 'fc8'])

    def _conv(x, W, b):
        x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def _pool(x):
        return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    def _fc(x, W, b):
        D = list(map(operator.attrgetter('value'), x.get_shape()))
        x = tf.reshape(x, (-1, np.prod(D[1:])))
        x = tf.matmul(x, W)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    net = {'input': input}
    weights = np.load('vgg16_weights.npz')

    x = input
    for layer in layers:
        try:
            W = weights[layer + '_W']
            b = weights[layer + '_b']
        except KeyError:
            pass

        if 'conv' in layer:
            x = _conv(x, W, b)
        elif 'pool' in layer:
            x = _pool(x)
        elif 'fc' in layer:
            x = _fc(x, W, b)
        net[layer] = x
    # net['score'] = tf.nn.softmax(x)

    return net

def gram(layer):
    H, W, C = layer.shape
    layer = layer.reshape((H * W, C))
    return layer.T.dot(layer)

def gram_tf(layer):
    _, H, W, C = layer.get_shape().as_list()
    layer = tf.reshape(layer, (H * W, C))
    return tf.matmul(layer, layer, transpose_a=True)

if __name__ == '__main__':
    img1 = scipy.misc.imread('2.jpg', mode='RGB')
    img1 = scipy.misc.imresize(img1, (224, 224))
    img1 = img1.astype(np.float32)
    img1 = preprocess(img1)

    shape = list(img1.shape)

    img2 = scipy.misc.imread('style2.jpg', mode='RGB')
    img2 = scipy.misc.imresize(img2, (224, 224))
    # img2 = scipy.misc.imresize(img2, shape[:2])
    img2 = img2.astype(np.float32)
    img2 = preprocess(img2)

    with tf.Session() as sess:
        imgs = tf.placeholder(tf.float32, [None] + shape)
        vgg = build_net(imgs)
        objs = [vgg[layer] for layer in [CONTENT_LAYER] + STYLE_LAYERS]
        objs = sess.run(
            objs,
            feed_dict={
                vgg['input']: [img1, img2]
            }
        )

        content = objs[0][0]
        styles = [s[1] for s in objs[1:]]
        gram = {name:gram(s) for name, s in zip(STYLE_LAYERS, styles)}


        img = tf.Variable(tf.truncated_normal([1] + shape, stddev=0.256))
        vgg = build_net(img)

        content_loss = 0.0
        content_loss = tf.nn.l2_loss(vgg[CONTENT_LAYER] - content) / tf.nn.l2_loss(content)
        style_loss = 0.0
        for layer in STYLE_LAYERS:
            size = reduce(operator.mul, vgg[layer].get_shape().as_list())
            style_loss += tf.nn.l2_loss(gram_tf(vgg[layer]) - gram[layer]) / len(STYLE_LAYERS) / tf.nn.l2_loss(gram[layer])
        _, H, W, _ = img.get_shape().as_list()
        B = 128
        size = reduce(operator.mul, img.get_shape().as_list())
        reg_loss = tf.nn.l2_loss(img) / (H * W * B ** 2)

        tv_loss = 2 * (tf.nn.l2_loss(img[:, 1:, :, :] - img[:, :H-1, :, :]) + tf.nn.l2_loss(img[:, :, 1:, :] - img[:, :, :W-1, :])) / (H * W * B ** 2)


        loss = (CONTENT_WEIGHT * content_loss +
                STYLE_WEIGHT * style_loss +
                REG_WEIGHT * reg_loss +
                TV_WEIGHT * tv_loss)
        train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        init = tf.initialize_all_variables()
        sess.run(init)

        best_loss = float('inf')
        best_image = None
        for i in range(ITERATIONS):
            c, s, t, l, _ = sess.run([content_loss, style_loss, tv_loss, loss, train])
            if l < best_loss:
                best_loss = l
                best_image = img.eval()[0]
            clipped = np.clip(img.eval(), -128.0, 127.0)
            img.assign(clipped)
            if not i % 50:
                print('Content: {} style: {} tv: {} total: {}'.format(c, s, t, l))
        print('Content: {} style: {} tv: {} total: {}'.format(c, s, t, l))

        img = np.clip(unprocess(best_image), 0, 255).astype(np.uint8)
        plt.imshow(img)
        plt.show()




