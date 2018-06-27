import tensorflow as tf

# define convolutional filter(for reusability)
# basic class for convolution...
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]

        conv = tf.concat(axis=3, values = output_groups)


    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    relu = tf.nn.relu(bias, name=scope.name)

    return relu

# define full connected layer
# basic class for sfc_sigmoid layer
def fc_sig(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:

        weights = tf.get_variable(shape=[num_in, num_out], trainable=True, name=name+'_weights')
        biases = tf.get_variable(shape=[num_out], trainable=True, name=name+'_biases')

        act = tf.nn.xw_plus_b(x, weights, biases, name=name+'_mul')

        sig = tf.nn.sigmoid(act, name=name+'_sig')

        return sig

# define full connected layer
# basic class for fc layer
def fc(x,num_in, num_out, name, relu = True):
    with tf.variable_scope(name) as scope:

        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

# define max pooling
# basic class for max pooling
def max_pool(x,filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

# define local response normalization
# basic class for local response normalization
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)