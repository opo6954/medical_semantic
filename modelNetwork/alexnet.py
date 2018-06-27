import tensorflow as tf
import numpy as np

import modelClass
import tfWrapper

'''
Alexnet class


'''
class AlexNet(modelClass.PreTrainedClassifyNet):
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

        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = tfWrapper.fc(dropout7, 4096, self.num_classes, relu=False, name='fc8')

    # load initial weight from the pre-trained weight file...
    def load_initial_weights(self, session):
        weights_dict = np.load(self.weights_path, encoding='bytes').item()
        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if(len(data.shape) == 1):
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))




