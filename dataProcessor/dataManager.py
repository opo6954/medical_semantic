import tensorflow as tf
import numpy as np
import cv2
import os


class DataManager:
    def __init__(self, imgNpyPath, labelNpyPath, batchSize, classNumber):
        self.img_data = np.load(imgNpyPath)
        self.label_data = np.load(labelNpyPath)
        self.dataSize = np.shape(self.img_data)[0]

        img_data_tmp = self.img_data
        label_data_tmp = self.label_data

        permutation = np.random.permutation(self.dataSize)

        self.img_data = []
        self.label_data = []
        for i in permutation:
            self.img_data.append(img_data_tmp[i])
            self.label_data.append(label_data_tmp[i])

        self.img_data = np.asarray(self.img_data)
        self.label_data = np.asarray(self.label_data)



        self.batchSize = batchSize
        self.classNumber = classNumber

        # mean value of imagenet dataset
        self.IMAGENET_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

        imgShape = np.shape(self.img_data)
        labelShape = np.shape(self.label_data)

        self.x = tf.placeholder(self.img_data.dtype, shape=[None, imgShape[1], imgShape[2], imgShape[3]])
        self.y = tf.placeholder(self.label_data.dtype, shape=[None, classNumber])


        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.dataset = self.dataset.batch(self.batchSize)

        self.iterator = self.dataset.make_initializable_iterator()

        self.next_input, self.next_label = self.iterator.get_next()

    def getNextBatchPlaceholderForNIH(self):
        input_value = tf.image.convert_image_dtype(self.next_input, tf.float32)

        input_value = input_value - self.IMAGENET_MEAN

        # const = tf.zeros(input_value.shape)
        # const = const - self.IMAGENET_MEAN

        # input_value = tf.add(input_value, const)

        return input_value, self.next_label
    def getNextBatchPlaceholder(self):

        input_value = tf.image.convert_image_dtype(self.next_input, tf.float32)

        return input_value, self.next_label

    def getTrueData(self):
        return self.img_data, self.label_data

    def getInitializer(self):
        return self.iterator.initializer
        # return self.img_data, self.label_data


# Sample code for dataManager and training

# trainData = DataManager('../dataNpyFiles/bbox_cropped_train_img.npy', '../dataNpyFiles/bbox_cropped_train_label.npy', 10, classNumber=6)






'''
classNumber = 15

trainData = DataManager(imgNpyPath='../../data_NIH/npyData/NIH_resize224_train_img.npy', labelNpyPath='../../data_NIH/npyData/NIH_resize224_train_label.npy', batchSize=32, classNumber=15)

input_value, label_value = trainData.getNextBatchPlaceholder()

input_value_flat = tf.reshape(input_value, [-1, 227 * 227 * 3])

net = tf.layers.dense(input_value_flat, 8)
prediction = tf.layers.dense(net, classNumber)

loss = tf.losses.mean_squared_error(prediction, label_value)
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    q = trainData.getTrueData()
    sess.run(trainData.getInitializer(), feed_dict={trainData.x: q[0], trainData.y: q[1]})

    for i in range(1000):
        _, loss_value = sess.run([train_op, loss])
        print('Iter: {}, Loss: {:.4f}'.format(i, loss_value))

'''

'''
EPOCH=20

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
dataset = tf.data.Dataset.from_tensor_slices((x,y))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(2)
dataset = dataset.repeat()

train_data = (np.random.sample((10,2)), np.random.sample((10,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))

print(train_data)

iter = dataset.make_initializable_iterator()

input_value, label_value = iter.get_next()



net = tf.layers.dense(input_value, 8)
prediction = tf.layers.dense(net, 1)

loss = tf.losses.mean_squared_error(prediction, label_value)
train_op = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iter.initializer, feed_dict={x: train_data[0], y: train_data[1]})

    for i in range(EPOCH):
        _, loss_value = sess.run([train_op, loss])
        print('Iter: {}, Loss: {:.4f}'.format(i, loss_value))
'''


'''
imgData = np.load('../dataNpyFiles/bbox_cropped_train_img.npy')
labelData = np.load('../dataNpyFiles/bbox_cropped_train_label.npy')

batchSize = 32

# imgData_placeholder = tf.placeholder(imgData.dtype, [batchSize, imgData.shape[1], imgData.shape[2], imgData.shape[3]])
# labelData_placeholder = tf.placeholder(labelData.dtype, [batchSize, 1])

imgData_placeholder = tf.placeholder(imgData.dtype, imgData.shape)
labelData_placeholder = tf.placeholder(labelData.dtype, labelData.shape)

dataset = tf.data.Dataset.from_tensor_slices((imgData_placeholder, labelData_placeholder))

dataset = dataset.shuffle(buffer_size=10000)

dataset = dataset.repeat()

# dataset = dataset.batch(32)
# dataset = dataset.repeat()


iterator = dataset.make_initializable_iterator()
imgs, labels = iterator.get_next()


with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={imgData_placeholder: imgData, labelData_placeholder: labelData})
    for k in range(10):
        print('epoch for ' + str(k))
        sess.run([imgs, labels])


'''


















