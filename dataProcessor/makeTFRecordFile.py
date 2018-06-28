import tensorflow as tf
import cv2
import numpy as np
import os

'''
save TFRecord file; serialize image and label together
NOT IMPLEMENTATION.......
'''

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

'''
Reconstruction from TFRecord save file
'''


def loadTFRecordReconFile2(tfRecordFilePath):
    def _parse(example_proto):
        features = {'image': tf.FixedLenFeature((), tf.string, default_value=""),
                    'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'width': tf.FixedLenFeature((), tf.int64, default_value=0)}

        parsed_features = tf.parse_single_example(example_proto, features)

        imgRaw = parsed_features['image']
        label = parsed_features['label']

        img = tf.decode_raw(imgRaw, tf.uint8)
        img.set_shape([3 * 224 * 224])

        img = tf.cast(tf.transpose(tf.reshape(img, [3, 224, 224]), [1,2,0]), tf.float32)


        return img, parsed_features['label']

    dataset = tf.data.TFRecordDataset(tfRecordFilePath)

    dataset = dataset.map(_parse)

    dataset = dataset.repeat()
    dataset = dataset.batch(32)

    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    sess = tf.Session()


    print('hello')


def loadTFRecordReconFile(tfRecordFilePath):

    reconstructed_img_list = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfRecordFilePath)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_string = (example.features.feature['image_raw'].bytes_list.value[0])
        label = (example.features.feature['label'].int64_list.value[0])


        img_1d = np.fromstring(img_string, dtype=np.uint8)

        image_decode = tf.image.decode_png(img_string, channels=3)

        image_decode = tf.cast(image_decode , tf.float32)

        # image_decode = tf.image.decode_image(img_string)



        # reconstructed_img = img_1d.reshape((height, width, -1))

        # cv2.imshow('power', reconstructed_img)
        with tf.Session() as sess:
            cv2.imshow('power', image_decode.eval())
            cv2.waitKey(0)




    return


'''
Save TFRecord type from image and label
'''
def saveTFRecordFile(dataLabelFilePath, outputFilePath, isResize = True, resizeFactor=224):
    writer = tf.python_io.TFRecordWriter(outputFilePath)



    with open(dataLabelFilePath) as readFile:
        allContents = readFile.readlines()

    for lineContent in allContents:
        print('load for ' + lineContent)
        tmpList = lineContent.split(',')
        imgPath = tmpList[0]
        imgLabel = tmpList[1]

        if(not os.path.exists(imgPath)):
            print('No image path ' + imgPath + ' exist...')
            continue

        img = cv2.imread(imgPath)

        if(isResize == True):
            img = cv2.resize(img,(resizeFactor, resizeFactor))


        height = img.shape[0]
        width = img.shape[1]
        label = int(imgLabel)

        img = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img),
            'label': _int64_feature(label)
        }))

        writer.write(example.SerializeToString())
    writer.close()


# saveTFRecordFile(dataLabelFilePath='../dataLabelFiles/wholeData_bbox_cropped_train.label', isResize=True, resizeFactor=224, outputFilePath='../dataTFRecordFiles/wholeData_bbox_cropped_resize224_train.tfrecords')
# saveTFRecordFile(dataLabelFilePath='../dataLabelFiles/wholeData_bbox_cropped_test.label', isResize=True, resizeFactor=224,  outputFilePath='../dataTFRecordFiles/wholeData_bbox_cropped_resize224_test.tfrecords')

# loadTFRecordReconFile(tfRecordFilePath='../dataTFRecordFiles/wholeData_bbox_cropped_resize224_train.tfrecords')