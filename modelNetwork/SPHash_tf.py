'''
Semantic Preserved Hash code
'''

import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import datetime
import modelClass
sys.path.insert(0, '../dataProcessor')

import dataManager

import alexnet
import tfWrapper



# first test for alexnet
# after that, porting SH net to tf(add few layers with loss)

# SPDH class that adding new layer with new loss function...

class SPDH_tf(modelClass.PreTrainedClassifyNet):
    def __init__(self, x_shape, y_shape, keep_prob, num_classes, skip_layer, batchSize, weights_path='', bitSize=16, learningRate=0.00001, weightSC=1, weightApp=1, weightFire=1):
        modelClass.PreTrainedClassifyNet.__init__(self, keep_prob=keep_prob, x_shape=x_shape, y_shape=y_shape, num_classes=num_classes, skip_layer=skip_layer, batchSize=batchSize, weights_path=weights_path)
        self.bitSize = bitSize
        self.weightSC = weightSC
        self.weightApp = weightApp
        self.weightFire = weightFire
        self.learningRate = learningRate

    def build_basic_graph(self):
        conv1 = tfWrapper.conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = tfWrapper.lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = tfWrapper.max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
        conv2 = tfWrapper.conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = tfWrapper.lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = tfWrapper.max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = tfWrapper.conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = tfWrapper.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = tfWrapper.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = tfWrapper.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = tfWrapper.fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = tfWrapper.dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = tfWrapper.fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = tfWrapper.dropout(fc7, self.keep_prob)

        # laten Layer: FC(w sigmoid)
        self.latentLayer = tfWrapper.fc_sig(dropout7, 4096, self.bitSize, name='latentLayer')

        self.softmaxLayer = tfWrapper.fc(self.latentLayer, self.bitSize, self.num_classes, relu=False, name='softmax')

    # load initiali weight
    def load_initial_weights(self, session):
        # Load the weights into memory
        weights_dict = np.load(self.weights_path, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.skip_layer:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


    def design_loss(self):

        with tf.name_scope("loss_cross_ent"):
            self.loss_sc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.softmaxLayer, labels=self.y))
        with tf.name_scope("loss_approach_0or1"):
            self.loss_app = -tf.reduce_mean(
                tf.subtract(self.latentLayer, 0.5 * tf.zeros(shape=tf.shape(self.latentLayer), dtype=tf.float32)))
        with tf.name_scope("loss_fire_0or1"):
            self.loss_fire = tf.subtract(tf.reduce_mean(self.latentLayer), 0.5)

        self.loss_total = self.weightSC * self.loss_sc + self.weightApp * self.loss_app + self.weightFire * self.loss_fire

        tf.summary.scalar('loss_cross_entropy', self.loss_sc)
        tf.summary.scalar('loss_approach', self.loss_app)
        tf.summary.scalar('loss_fire', self.loss_fire)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss_total)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.softmaxLayer, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)

        self.merged_summary = tf.summary.merge_all()






