import tensorflow as tf

from definitions import DL_PATCH_HEIGHT, DL_PATCH_WIDTH, DL_LABEL_NR


class Model:
    """
    Class that defines U-Net graph.
    """
    def __init__(self):
        self._input_shape = [-1, DL_PATCH_HEIGHT, DL_PATCH_WIDTH, 1]
        self.conv1_1 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv1_2 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv2_1 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv2_2 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv3_1 = tf.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv3_2 = tf.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pool3 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv4_1 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv4_2 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pool4 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv5_1 = tf.layers.Conv2D(filters=1024, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv5_2 = tf.layers.Conv2D(filters=1024, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.pool5 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        self.conv6 = tf.layers.Conv2D(filters=1024, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv7_1 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv_transpose0 = tf.layers.Conv2DTranspose(filters=512, kernel_size=[3, 3], strides=[2, 2], padding='same')
        self.conv7_2 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv8_1 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv_transpose1 = tf.layers.Conv2DTranspose(filters=512, kernel_size=[3, 3], strides=[2, 2], padding='same')
        self.conv8_2 = tf.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv9_1 = tf.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv_transpose2 = tf.layers.Conv2DTranspose(filters=256, kernel_size=[3, 3], strides=[2, 2], padding='same')
        self.conv9_2 = tf.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv10_1 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv_transpose3 = tf.layers.Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same')
        self.conv10_2 = tf.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv11_1 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv_transpose4 = tf.layers.Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=[2, 2], padding='same')
        self.conv11_2 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv12 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.layers.Conv2D(filters=DL_LABEL_NR, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, features):
        image = features['image']
        y = tf.reshape(image, self._input_shape)
        y = self.conv1_1(y)
        conv1_2 = self.conv1_2(y)
        y = self.pool1(conv1_2)
        y = self.conv2_1(y)
        conv2_2 = self.conv2_2(y)
        y = self.pool2(conv2_2)
        y = self.conv3_1(y)
        conv3_2 = self.conv3_2(y)
        y = self.pool3(conv3_2)
        y = self.conv4_1(y)
        conv4_2 = self.conv4_2(y)
        y = self.pool4(conv4_2)
        y = self.conv5_1(y)
        conv5_2 = self.conv5_2(y)
        y = self.pool5(conv5_2)
        y = self.conv6(y)
        y = self.conv7_1(y)
        y = self.conv_transpose0(y)
        y = tf.concat([y, conv5_2], axis=3)
        y = self.conv7_2(y)
        y = self.conv8_1(y)
        y = self.conv_transpose1(y)
        y = tf.concat([y, conv4_2], axis=3)
        y = self.conv8_2(y)
        y = self.conv9_1(y)
        y = self.conv_transpose2(y)
        y = tf.concat([y, conv3_2], axis=3)
        y = self.conv9_2(y)
        y = self.conv10_1(y)
        y = self.conv_transpose3(y)
        y = tf.concat([y, conv2_2], axis=3)
        y = self.conv10_2(y)
        y = self.conv11_1(y)
        y = self.conv_transpose4(y)
        y = tf.concat([y, conv1_2], axis=3)
        y = self.conv11_2(y)
        y = self.conv12(y)
        return self.logits(y)
