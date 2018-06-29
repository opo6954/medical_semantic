README of PACS medical

Environment:
Tensorflow 1.7(lower is okay(1.x))
OpenCV 3.4.0.12(install with pip)



1. execution of sh file to get the large size data
--> github cannot push large size data(> 50 MB)
- sh get_ASAN_data_png_set.sh
--> get the original version of ASAN dataset ~ 19.7 GB

- sh get_ASAN_data_set.sh
--> get the label and npy data of ASAN dataset ~ 8.8 GB

- sh get_cifar10_data_set.sh
--> get the cifar10 dataset(general image data) ~ 37.8 GB
- sh get_init_model.sh
--> get the initial Alexnet pretrained model data ~ 243.9 MB

- sh get_trained_model.sh
--> get the sample model trained by SPDH net ~ 2.8 GB


2. train with ASAN dataset
train SPDH Net with ASAN dataset including npy file on testTrainModel.py
trained model is stored in largeData/model for default
label is 0,1,2,3,4,5 for nodule, consolidation, interstitialOpacity, Cardiomegaly, PleuralEffusion


3. test with SPDH Net
test(bit generator) with ASAN dataset including label and original image file on testAllTrainedModel.py
result is stored in bitStore folder for default
output format with fileName, label, bit representation
