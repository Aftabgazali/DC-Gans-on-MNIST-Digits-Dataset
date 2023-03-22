# DC-Gans-on-MNIST-Digits-Dataset
# Problem Statement: To Implement a DC-GAN Model that is capable of generating fake images of Hand-Written Digits.

## Introduction 
**Gans stands for Generative Adversarial Networks.** It is a type of deep learning model that is composed of two neural networks: a generator and a discriminator. The generator creates new data samples, while the discriminator attempts to distinguish the generated samples from real samples. The two networks are trained simultaneously, with the generator trying to create samples that can fool the discriminator, and the discriminator tries to correctly identify the generated samples. GANs have been used for a variety of tasks, including image synthesis, text generation, and anomaly detection.

## Dataset Introduction
A handwritten digits dataset is a collection of images of handwritten digits, along with their corresponding labels (i.e., the correct digit). These datasets are commonly used for training and evaluating machine learning algorithms, particularly in the field of image recognition and computer vision. One of the most popular datasets of this type is the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits, along with their corresponding labels. We are going to use this dataset for training our GAN Model. The images in this dataset are 28x28 pixels and grayscale images, and it is widely used as a benchmark dataset for testing and comparing the performance of different machine learning algorithms. Other examples of handwritten digits datasets include the USPS dataset, which contains images of USPS postal codes, and the KMNIST dataset, which is a more diverse set of handwritten digits than the MNIST dataset.

![image](https://user-images.githubusercontent.com/52740449/226985487-719c16fb-af4e-4250-ae13-6abe66138725.png)
                                                 Sample Handwritten Dataset
                                                 
## 2 Dataset Preprocessing
First and foremost step for every Machine Learning Problem, is to preprocess the given dataset. Preprocessing the data would ensure that the learning accuracy of model is increased and the prediction capability of model is improved.
### 2.1 
