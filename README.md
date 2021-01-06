This repository contains the source code for developing a deep learning (DL) system for the automated classification of keratitis, other cornea abnormalities, and normal cornea from slit-lamp and smartphone images. 
This system has the potential to be applied to both digital slit lamp cameras and smartphones to promote the early diagnosis and treatment of keratitis, preventing the corneal blindness caused by keratitis.

* Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Jiewei Jiang, E-mail: jiangjw924@126.com.

* The representative samples for keratitis, other cornea abnormalities, and normal cornea are presented in /Keratitis-Source/sample.

* All codes were executed in the Pytorch (1.6.0 above) framework with Ubuntu 18.04 64bit + CUDA (Compute Unified Device Architecture).

* The file "keratitis_training_v1.py" in /Keratitis-Source is used for network training.

* The file "keratitis_testing_v1.py" in /Keratitis-Source is used for testing.

* The training and testing are executed as follows:

# Train DenseNet121 on GPU
python keratitis_training_v1.py -a 'densenet121'  

# Train ResNet50 on GPU
python keratitis_training_v1.py -a 'resnet50'  

# Train Inception-v3 on GPU
python keratitis_training_v1.py -a 'inception_v3'  

# Evaluate three models of DenseNet121, ResNet50, and Inception-v3 at the same time on GPU
python keratitis_testing_v1.py
