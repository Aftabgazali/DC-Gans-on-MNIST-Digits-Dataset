# DC-Gans-on-MNIST-Digits-Dataset
# Problem Statement: To Implement a DC-GAN Model that is capable of generating fake images of Hand-Written Digits.

## 1. Introduction 
**GANs stands for Generative Adversarial Networks.** It is a type of deep learning model that is composed of two neural networks: a generator and a discriminator. The generator creates new data samples, while the discriminator attempts to distinguish the generated samples from real samples. The two networks are trained simultaneously, with the generator trying to create samples that can fool the discriminator, and the discriminator tries to correctly identify the generated samples. GANs have been used for a variety of tasks, including image synthesis, text generation, and anomaly detection.

## 1.1 Dataset Introduction
A handwritten digits dataset is a collection of images of handwritten digits, along with their corresponding labels (i.e., the correct digit). These datasets are commonly used for training and evaluating machine learning algorithms, particularly in the field of image recognition and computer vision. One of the most popular datasets of this type is the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits, along with their corresponding labels. We are going to use this dataset for training our GAN Model. The images in this dataset are 28x28 pixels and grayscale images, and it is widely used as a benchmark dataset for testing and comparing the performance of different machine learning algorithms. Other examples of handwritten digits datasets include the USPS dataset, which contains images of USPS postal codes, and the KMNIST dataset, which is a more diverse set of handwritten digits than the MNIST dataset.

![image](https://user-images.githubusercontent.com/52740449/226985487-719c16fb-af4e-4250-ae13-6abe66138725.png)
                                                
                                                   Handwritten Dataset
                                                 
## 2 Dataset Preprocessing
First and foremost step for every Machine Learning Problem, is to preprocess the given dataset. Preprocessing the data would ensure that the learning accuracy of model is increased and the prediction capability of model is improved.
### 2.1 Feature Scaling
Feature Scaling is an important step of data preprocessing, If feature scaling is not performed, a machine learning algorithm would consider larger values to be higher and smaller values to be lower, regardless of the unit of measurement. The scaler used here is the Traditional MinMax Scaler. Minmax scaler Transform features by scaling each feature to a given range[0,1]
### 2.2 Expand Dims
In the python NumPy library, numpy.expand_dims is a function that inserts a new axis with size 1.
### 2.3 Labeling Real & Fake Samples
Before Building GAN Model, we have to prepare the data properly, first we've to label actual data which is real or which ones are fake(0,1). Real Samples are taken within a random range say (0-500). While for fake samples we generate a random latent variable reshape to form an image and label it as '0'.

## 3. Working of GANs

![image](https://user-images.githubusercontent.com/52740449/227017401-729b867a-2fd2-40dd-bc78-3f95d69592c8.png)

Gans consists of two models that is competing simultaneously with each other: a generator and a discriminator. As discussed above the generator takes in a random latent sample and generates the image this image is then passed to the discriminator which is trained with the training dataset and hence already knows which images are real and which images are fake. The task of the discriminator is to identify whether the generated image is either real or fake. The two networks are trained simultaneously, with the generator trying to produce synthetic data that can fool the discriminator, and the discriminator tries to correctly identify whether each piece of data is real or fake.

As training progresses, the generator improves at producing synthetic data that looks more and more like real-world data, while the discriminator becomes better at distinguishing the synthetic data from the real data.

## 3.1 Building the Discriminator Model
We have build a deeply connected discriminator model with 2 convolutional layers and connected it with final output layer, we have given strides of (2,2) to downsample the image, padding is kept same to keep the shape same. Each layer is fitted with a dropout to avoid overfitting. In short discriminator is nothing but a simple classifier CNN model which takes an Image and gives back label wheter 0 or 1.

## 3.2 Building the Generator Model
Generator is built different from the discriminator, we upsample the image which we downsampled it in discriminator this is done with the help of combined function called 'Con2DTranspose'. As we know generator's job is to take any random latent variable and give back a generated image. Hence we start of by taking a random image foundation of 128 x 7 x 7. 

## 3.3 Combined GAN Model.
Due to its Adversarial Nature, Training Generator and Discriminator is not easy and takes a long time. However using few gimmicks can help us overcome the time barrier. Combining both Generator and Discriminator in one Model and training them using few tricks would allow us to build a perfect GAN model. In-short first we are going to train our discriminator model, then the output which was given by discriminator is used to train the Generator. 
Finally after 100 epochs we will be able to see results generated by our Generator.

## 4. Results
Training GANs takes a lot of time and consume a lot of gpu power, this project was done on google-colab due to only given acess to gpu for a short amount of resource, training more than 150 epochs is almost impossible in free version of google-colab. However still by training it to only 100 epochs we were able to achive good results.

![image](https://user-images.githubusercontent.com/52740449/227020978-33bcb539-47bf-4e1f-99c5-e44d06154dff.png)


# For More in-depth explanation please visit my website
https://aftabgazali001.medium.com/gans-on-hand-written-digits-dataset-371e4f46da93

