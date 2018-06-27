
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import datetime
import modelClass
sys.path.insert(0, '../dataProcessor')
sys.path.insert(0, '../Alexnet/finetune_alexnet_with_tensorflow')
from datagenerator import ImageDataGenerator
import datagenerator

import dataManager
import dataManagerFromDisk





from SPHash_tf import SPDH_tf

import alexnet
import tfWrapper


savor_max_to_keep = 5
model_save_step = 5


def train_SPHash_from_dataGenerator():
    prefix = 'SPDH_dataGenerator'

    batchSize = 32
    learning_rate = 0.0001
    num_epoches = 30
    display_step = 20
    dropout_rate = 0.5

    num_classes = 6

    filewriter_path = '../modelStore/ASAN/tensorboardPath'
    checkpoint_path = '../modelStore/ASAN/modelTrainedPath'

    train_file = '../dataOpenFormat/train.txt'
    val_file = '../dataOpenFormat/test.txt'

    with tf.device('/cpu:0'):
        print('load train data...')
        tr_data = ImageDataGenerator(txt_file=train_file, mode='training', batch_size=batchSize, num_classes=num_classes, shuffle=True)

        print('load test data...')
        val_data = ImageDataGenerator(txt_file=val_file, mode='inference', batch_size=batchSize, num_classes=num_classes, shuffle=False)

        iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
        iterator_val = tf.data.Iterator.from_structure(val_data.data.output_types, val_data.data.output_shapes)

        next_batch = iterator.get_next()
        next_batch_val = iterator_val.get_next()

    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator_val.make_initializer(val_data.data)







    x = tf.placeholder(tf.float32, [batchSize, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batchSize, num_classes])

    # TF placeholder for graph input and output

    keep_prob = tf.placeholder(tf.float32)
    train_layers = ['fc8', 'fc7', 'fc6']

    # Initialize model
    model = SPDH_tf(x_shape=x, y_shape=y, keep_prob=keep_prob, num_classes=num_classes, skip_layer=train_layers, weights_path='../Alexnet/bvlc_alexnet.npy', batchSize=batchSize)

    model.build_basic_graph()
    model.design_loss()

    writer = tf.summary.FileWriter(filewriter_path)
    saver = tf.train.Saver(max_to_keep=savor_max_to_keep)

    train_batches_per_epoch = int(np.floor(tr_data.data_size / batchSize))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batchSize))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        # for fine-tuning
        model.load_initial_weights(sess)


        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(),
                                                          filewriter_path))



        for epoch in range(num_epoches):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))

            sess.run(training_init_op)

            for step in range(train_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)

                lossTotal, lossSC, lossApp, lossFire, _ = sess.run(
                    [model.loss_total, model.loss_sc, model.loss_app, model.loss_fire, model.train_op],
                    feed_dict={model.x: img_batch, model.y: label_batch, model.keep_prob: dropout_rate})

                if step % display_step == 0:
                    s = sess.run(model.merged_summary, feed_dict={model.x: img_batch,
                                                                  model.y: label_batch,
                                                                  model.keep_prob: 1.})

                    print('At ' + str(step) + ', train loss: ' + str(lossTotal) + ', train SC: ' + str(
                        lossSC) + ', train App: ' + str(lossApp) + ', trainFire: ' + str(lossFire))

                    writer.add_summary(s, epoch * train_batches_per_epoch + step)

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.datetime.now()))

            print('testData initializer on GPU')

            sess.run(validation_init_op)
            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch_val)

                acc = sess.run(model.accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                test_acc += acc
                test_count += 1


            test_acc /= test_count

            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, prefix + 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(),
                                                           checkpoint_name))



def train_SPHash_cifar10(modelSavedPath = '../modelStore/ASAN', trainLabel_path='', testLabel_path='', bitSize=16, num_epochs=30, weightSM=1.0, weightApp=1.0, weightFire=1.0):
    tf.reset_default_graph()

    prefix = 'cifar10' + '_' + str(bitSize) + '_wSM_' + str(weightSM) + '_wApp_' + str(
        weightApp) + '_wFire_' + str(weightFire)

    batchSize = 32
    dropout_rate=0.5
    display_step = 20

    num_classes = 10

    filewriter_path = modelSavedPath + '/tensorboardPath'
    checkpoint_path = modelSavedPath + '/modelTrainedPath'

    if(not os.path.exists(filewriter_path)):
        os.mkdir(filewriter_path)
    if(not os.path.exists(checkpoint_path)):
        os.mkdir(checkpoint_path)



    print('load train data...')

    trainData = dataManagerFromDisk.DataManagerFromDisk(labelFilePath=trainLabel_path, batchSize=batchSize,
                                        totalClass=num_classes)
    print('load test data...')
    testData = dataManagerFromDisk.DataManagerFromDisk(labelFilePath=testLabel_path, batchSize=batchSize,
                                       totalClass=num_classes)

    trainDataSize = trainData.getLengthOfAllImg()
    testDataSize = testData.getLengthOfAllImg()

    x = tf.placeholder(tf.float32, [batchSize, 227, 227, 3], name='input_network')
    y = tf.placeholder(tf.float32, [batchSize, num_classes], name='label_network')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')



    model = SPDH_tf(bitSize=bitSize, x_shape=x, y_shape=y, keep_prob=keep_prob, num_classes=num_classes,
                    skip_layer=['fc8', 'fc7', 'fc6'], weights_path='../Alexnet/bvlc_alexnet.npy',
                    batchSize=batchSize, weightSC=weightSM, weightApp=weightApp, weightFire=weightFire)



    model.build_basic_graph()
    model.design_loss()

    writer = tf.summary.FileWriter(filewriter_path)
    saver = tf.train.Saver(max_to_keep=savor_max_to_keep)

    train_batches_per_epoch = int(np.floor(trainDataSize / batchSize))
    val_batches_per_epoch = int(np.floor(testDataSize / batchSize))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        writer.add_graph(sess.graph)

        model.load_initial_weights(sess)

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):


            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))

            for step in range(train_batches_per_epoch):

                # And run the training op
                # trainLoss, _ = sess.run([loss, optimizer], feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})

                # for TEST

                nextBatch = trainData.returnNextBatch()
                image_batch = nextBatch[0]
                label_batch = nextBatch[1]

                lossTotal, lossSC, lossApp, lossFire, _ = sess.run(
                    [model.loss_total, model.loss_sc, model.loss_app, model.loss_fire, model.train_op],
                    feed_dict={model.x: image_batch, model.y: label_batch, model.keep_prob: dropout_rate})

                # Generate summary with the current
                #
                # cat batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(model.merged_summary, feed_dict={model.x: nextBatch[0],
                                                                  model.y: nextBatch[1],
                                                                  model.keep_prob: 1.})
                    print('At ' + str(step) + ', train loss: ' + str(lossTotal) + ', train SC: ' + str(
                        lossSC) + ', train App: ' + str(lossApp) + ', trainFire: ' + str(lossFire))

                    writer.add_summary(s, epoch * train_batches_per_epoch + step)

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.datetime.now()))

            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                nextBatch = testData.returnNextBatch()
                img_batch = nextBatch[0]
                label_batch = nextBatch[1]


                acc = sess.run(model.accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count

            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            if (epoch % model_save_step == 0):
                checkpoint_name = os.path.join(checkpoint_path, prefix + 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_name))


def train_SPHash(modelSavedPath = '../modelStore/ASAN', inputType='Cropped', bitSize=16, num_epochs=30, isJH = False, weightSM = 1.0, weightApp = 1.0, weightFire = 1.0):
    tf.reset_default_graph()

    if(isJH  == True):
        prefix = 'JH_SPDH' + inputType + '_' + str(bitSize) + '_wSM_' + str(weightSM) + '_wApp_' + str(weightApp) + '_wFire_' + str(weightFire)
    else:
        prefix = 'SPDH' + inputType + '_' + str(bitSize) + '_wSM_' + str(weightSM) + '_wApp_' + str(weightApp) + '_wFire_' + str(weightFire)

    batchSize = 32
    dropout_rate = 0.5
    display_step = 20

    # For ASAN Data

    num_classes = 6

    # filewriter_path = '../modelStore/ASAN/tensorboardPath'
    # checkpoint_path = '../modelStore/ASAN/modelTrainedPath'

    filewriter_path = modelSavedPath + '/tensorboardPath'
    checkpoint_path = modelSavedPath + '/modelTrainedPath'

    if(not os.path.exists(filewriter_path)):
        os.mkdir(filewriter_path)
    if(not os.path.exists(checkpoint_path)):
        os.mkdir(checkpoint_path)

    if(isJH == False):
        if(inputType == 'Cropped'):
            trainImg_path = '../dataNpyFiles/bbox_cropped_train_img.npy'
            trainLabel_path = '../dataNpyFiles/bbox_cropped_train_label.npy'
            testImg_path = '../dataNpyFiles/bbox_cropped_test_img.npy'
            testLabel_path = '../dataNpyFiles/bbox_cropped_test_label.npy'
        elif(inputType == 'Activated'):
            trainImg_path = '../dataNpyFiles/bbox_activated_train_img.npy'
            trainLabel_path = '../dataNpyFiles/bbox_activated_train_label.npy'
            testImg_path = '../dataNpyFiles/bbox_activated_test_img.npy'
            testLabel_path = '../dataNpyFiles/bbox_activated_test_label.npy'
        elif(inputType == 'Origin'):
            trainImg_path = '../dataNpyFiles/origin_train_img.npy'
            trainLabel_path = '../dataNpyFiles/origin_train_label.npy'
            testImg_path = '../dataNpyFiles/origin_test_img.npy'
            testLabel_path = '../dataNpyFiles/origin_test_label.npy'
        else:
            print ('No available option found for ' + inputType)
            return
    else:
        if (inputType == 'Cropped'):
            trainImg_path = '../exp_jh/bbox_search_img.npy'
            trainLabel_path = '../exp_jh/bbox_search_label.npy'
            testImg_path = '../exp_jh/bbox_query_img.npy'
            testLabel_path = '../exp_jh/bbox_query_label.npy'
        elif (inputType == 'Activated'):
            trainImg_path = '../exp_jh/activated_search_img.npy'
            trainLabel_path = '../exp_jh/activated_search_label.npy'
            testImg_path = '../exp_jh/activated_query_img.npy'
            testLabel_path = '../exp_jh/activated_query_label.npy'
        elif (inputType == 'Origin'):
            trainImg_path = '../exp_jh/origin_search_img.npy'
            trainLabel_path = '../exp_jh/origin_search_label.npy'
            testImg_path = '../exp_jh/origin_query_img.npy'
            testLabel_path = '../exp_jh/origin_query_label.npy'
        else:
            print ('No available option found for ' + inputType)
            return

    print('load train data...')
    trainData = dataManager.DataManager(imgNpyPath=trainImg_path, labelNpyPath=trainLabel_path, batchSize=batchSize, classNumber=num_classes)
    print('load test data...')
    testData = dataManager.DataManager(imgNpyPath=testImg_path, labelNpyPath=testLabel_path, batchSize=batchSize, classNumber=num_classes)


    trainDataSize = np.shape(trainData.label_data)[0]
    testDataSize = np.shape(testData.label_data)[0]

    x = tf.placeholder(tf.float32, [batchSize, 227, 227, 3], name='input_network')
    y = tf.placeholder(tf.float32, [batchSize, num_classes], name='label_network')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')



    model = SPDH_tf(bitSize=bitSize, x_shape=x, y_shape=y, keep_prob=keep_prob, num_classes=num_classes,
                    skip_layer=['fc8', 'fc7', 'fc6'], weights_path='../Alexnet/bvlc_alexnet.npy',
                    batchSize=batchSize, weightSC=weightSM, weightApp=weightApp, weightFire=weightFire)


    model.build_basic_graph()
    model.design_loss()

    writer = tf.summary.FileWriter(filewriter_path)
    saver = tf.train.Saver(max_to_keep=savor_max_to_keep)

    train_batches_per_epoch = int(np.floor(trainDataSize / batchSize))
    val_batches_per_epoch = int(np.floor(testDataSize / batchSize))



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        writer.add_graph(sess.graph)

        model.load_initial_weights(sess)

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(),
                                                          filewriter_path))





        # Loop over number of epochs
        for epoch in range(num_epochs):

            print('trainData initializer on GPU')
            q = trainData.getTrueData()
            sess.run(trainData.getInitializer(), feed_dict={trainData.x: q[0], trainData.y: q[1]})

            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))



            for step in range(train_batches_per_epoch):

                input_value, label_value = sess.run(trainData.getNextBatchPlaceholder())

                # And run the training op
                # trainLoss, _ = sess.run([loss, optimizer], feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})

                # for TEST
                p = sess.run(model.latentLayer, feed_dict={model.x: input_value, model.keep_prob: 1.0})





                lossTotal, lossSC, lossApp, lossFire, _ = sess.run([model.loss_total, model.loss_sc, model.loss_app, model.loss_fire, model.train_op],
                                        feed_dict={model.x: input_value, model.y: label_value, model.keep_prob: dropout_rate})
                # trainLoss = sess.run(train_op, feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})

                # Generate summary with the current
                #
                # cat batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(model.merged_summary, feed_dict={model.x: input_value,
                                                            model.y: label_value,
                                                            model.keep_prob: 1.})
                    print('At ' + str(step) + ', train loss: ' + str(lossTotal) + ', train SC: ' + str(lossSC) + ', train App: ' + str(lossApp) + ', trainFire: ' + str(lossFire))

                    writer.add_summary(s, epoch * train_batches_per_epoch + step)


            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.datetime.now()))

            test_acc = 0.
            test_count = 0

            print('testData initializer on GPU')
            p = testData.getTrueData()
            sess.run(testData.getInitializer(), feed_dict={testData.x: p[0], testData.y: p[1]})



            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(testData.getNextBatchPlaceholder())

                acc = sess.run(model.accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count





            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            if(epoch % model_save_step == 0):
                checkpoint_name = os.path.join(checkpoint_path, prefix + 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_name))

def test_pretrained(isNIH = False):

    batchSize = 32
    learning_rate= 0.0001
    num_epochs = 100
    dropout_rate = 0.5
    display_step = 20

    prefix = ''




    # For ASAN Data

    # num_classes = 6
    # filewriter_path = '../modelStore/ASAN/tensorboardPath'
    # checkpoint_path = '../modelStore/ASAN/modelTrainedPath'
    # trainImg_path = '../dataNpyFiles/bbox_cropped_train_img.npy'
    # trainLabel_path = '../dataNpyFiles/bbox_cropped_train_label.npy'
    # testImg_path = '../dataNpyFiles/bbox_cropped_test_img.npy'
    # testLabel_path = '../dataNpyFiles/bbox_cropped_test_label.npy'
    #
    # # For NIH Data
    if(isNIH == True):
        num_classes = 15
        filewriter_path = '../modelStore/NIH/tensorboardPath'
        checkpoint_path = '../modelStore/NIH/modelTrainedPath'
        trainImg_path = '../../data_NIH/npyData/train_val_list.txt_NIH_resize227_img.npy'
        trainLabel_path = '../../data_NIH/npyData/train_val_list.txt_NIH_resize227_label.npy'
        testImg_path = '../../data_NIH/npyData/test_list.txt_NIH_resize227_img.npy'
        testLabel_path = '../../data_NIH/npyData/test_list.txt_NIH_resize227_label.npy'
        prefix = 'NIH'
    else:
        num_classes = 6
        filewriter_path = '../modelStore/ASAN/tensorboardPath'
        checkpoint_path = '../modelStore/ASAN/modelTrainedPath'
        trainImg_path = '../dataNpyFiles/bbox_cropped_train_img.npy'
        trainLabel_path = '../dataNpyFiles/bbox_cropped_train_label.npy'
        testImg_path = '../dataNpyFiles/bbox_cropped_test_img.npy'
        testLabel_path = '../dataNpyFiles/bbox_cropped_test_label.npy'
        prefix = 'ASAN'


    print('load train data...')
    trainData = dataManager.DataManager(imgNpyPath=trainImg_path, labelNpyPath=trainLabel_path,
                                        batchSize=batchSize, classNumber=num_classes)
    print('load test data...')
    testData = dataManager.DataManager(imgNpyPath=testImg_path, labelNpyPath=testLabel_path,
                                       batchSize=batchSize, classNumber=num_classes)








    x = tf.placeholder(tf.float32, [batchSize, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batchSize, num_classes])

    # TF placeholder for graph input and output

    keep_prob = tf.placeholder(tf.float32)
    train_layers = ['fc8', 'fc7', 'fc6']

    # Initialize model
    model = alexnet.AlexNet(x_shape=x, y_shape=y, keep_prob=keep_prob, num_classes=num_classes, skip_layer=train_layers, weights_path='../Alexnet/bvlc_alexnet.npy', batchSize=batchSize)
    model.build_basic_graph()

    # Link variable to model output
    score = model.fc8



    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,labels=y))
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=y))

    # Train op
    with tf.name_scope("train"):

        # Create optimizer and apply gradient descent to the trainable variables
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        # correct_pred = tf.equal(tf.round(score), y)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        # accuracy for softmax label classifier
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=savor_max_to_keep)

    # Get the number of training/validation steps per epoch

    train_batches_per_epoch = int(np.floor(trainData.dataSize / batchSize))
    val_batches_per_epoch = int(np.floor(testData.dataSize / batchSize))

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)


        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(),
                                                          filewriter_path))

        sess.run(tf.global_variables_initializer())




        # Loop over number of epochs
        for epoch in range(num_epochs):

            print('trainData initializer on GPU')
            q = trainData.getTrueData()
            sess.run(trainData.getInitializer(), feed_dict={trainData.x: q[0], trainData.y: q[1]})

            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))



            for step in range(train_batches_per_epoch):

                # input_value, label_value = sess.run(trainData.getNextBatchPlaceholder())
                if(isNIH == True):
                    input_value, label_value = sess.run(trainData.getNextBatchPlaceholderForNIH())
                else:
                    input_value, label_value = sess.run(trainData.getNextBatchPlaceholder())

                # And run the training op
                # trainLoss, _ = sess.run([loss, optimizer], feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})
                trainLoss, _ = sess.run([loss, train_op], feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})
                # trainLoss = sess.run(train_op, feed_dict={x: input_value,y: label_value,keep_prob: dropout_rate})

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: input_value,
                                                            y: label_value,
                                                            keep_prob: 1.})
                    print('At ' + str(step) + ', train loss: ' + str(trainLoss))


                    writer.add_summary(s, epoch * train_batches_per_epoch + step)



            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.datetime.now()))

            test_acc = 0.
            test_count = 0

            print('testData initializer on GPU')
            p = testData.getTrueData()
            sess.run(testData.getInitializer(), feed_dict={testData.x: p[0], testData.y: p[1]})



            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(testData.getNextBatchPlaceholder())

                acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           prefix + 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(),
                                                           checkpoint_name))


# input processor as dataGenerator class(from open source)
# train_SPHash_from_dataGenerator()

# input processor as dataManager class(impl by me)

'''
train_SPHash(inputType="Cropped", bitSize=32, num_epochs=10)
train_SPHash(inputType="Activated", bitSize=32, num_epochs=10)
train_SPHash(inputType="Origin", bitSize=32, num_epochs=20)


train_SPHash(inputType="Cropped", bitSize=48, num_epochs=20)
train_SPHash(inputType="Activated", bitSize=48, num_epochs=20)
train_SPHash(inputType="Origin", bitSize=48, num_epochs=500)

train_SPHash(inputType="Cropped", bitSize=16, num_epochs=20)
train_SPHash(inputType="Activated", bitSize=16, num_epochs=20)
train_SPHash(inputType="Origin", bitSize=16, num_epochs=50)
'''
# test_pretrained(isNIH=False)