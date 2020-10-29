# BrainStation Capstone Project
## Driver Behavior Detection with Convolutional Neural Network
In this project, I structured a CNN framework using machine learning and transfer learning techniques to process image features. 
The object of this project is to build a machine learning model that can recognize 5 different behavior categories. 
When the distracted driving is happening, the model is able to remind the driver to prevent traffic accidents.

## Data Source :
The dataset comes from the [Kaggle](https://www.kaggle.com/robinreni/revitsone-5class) called ‘Driver Behavior Dataset’.
The dataset consists of 5 categories:
  - saft_driving
  - talking_phone
  - texting_phone
  - turning
  - other_activities
There are 10766 images in the format of JPG and PNG all across the 5 categories.

## Data Preprocessing & EDA:
  - Filter out the corrupted files: to prevent null values
  - Check data distribution: to prevent class imbalance(class imbalance will make model more capable of recognizing the class with large portion of data)
  - Resized all the images: tensorflow model requires all the images being the same size
  - Reconstruct the data set: split it into train set and test set, using test set to provide a totally unbiased estimate of the model's performance. This is data the model has never seen; it should serve as a good predictor for the model's performance once deployed and making prediction on new data.

After cleaning and reconstruction, we have a balanced data set without any null value:

![image](https://github.com/mengfei-liu/Capstone/blob/master/img/1.png)

There're 8600 pictures in 5 categories in the dataset.

- 1695 pictures in other_activities category.

- 1762 pictures in safe_driving category.

- 1735 pictures in talking_phone category.

- 1762 pictures in texting_phone category.

- 1646 pictures in turning category.



## Modeling

Epochs: 10
13 models were built to compare:

|                Model                 |   Type    | Train Accuracy | Validation Accuracy | Test Accuracy |
| :----------------------------------: | :-------: | :------------: | :-----------------: | :-----------: |
|         MobileNet_Benchmark          | Benchmark |     78.96%     |       72.12%        |    76.72%     |
|        MobileNetV2_Benchmark         | Benchmark |     77.89%     |       71.18%        |    75.98%     |
|           VGG16_Benchmark            | Benchmark |     62.77%     |       58.37%        |    63.27%     |
|           VGG19_Benchmark            | Benchmark |     61.15%     |       55.80%        |    62.29%     |
|          MobileNet_3Layers           | 3 Layers  |     96.97%     |       88.28%        |    95.53%     |
|         MobileNetV2_3Layers          | 3 Layers  |     97.30%     |       85.75%        |    94.51%     |
|            VGG16_3Layers             | 3 Layers  |     89.17%     |       84.77%        |    91.11%     |
|            VGG19_3Layers             | 3 Layers  |     87.31%     |       82.79%        |    88.64%     |
|          MobileNet_5Layers           | 5 Layers  |     98.12%     |       88.51%        |    95.72%     |
|         MobileNetV2_5Layers          | 5 Layers  |     97.89%     |       86.84%        |    95.39%     |
|        MobileNet_TensorBoard         | 3 Layers  |     98.29%     |       88.16%        |    94.73%     |
|      MobileNet_TensorBoard_AUG       | 3 Layers  |     52.15%     |       53.89%        |    59.73%     |
| MobileNet_TensorBoard_Pretrained_AUG | 3 Layers  |     76.26%     |       80.18%        |    90.83%     |

Used benchmark models and models with 3 dense layers and drop to compare 4 kinds of pre-trained model. Due to VGG models having the lower accuracy and longer prediction time, we kept the MobileNet and MobileNetV2 for further investigation. After added 5 layers to the benchmark models, the models’ performance only improved a little(less than 1%). To avoid extending the complexity, models with 3 dense layers and dropout are the best options. After tried other techniques, such as TensorBoard hyperparameter searching and image augmentation, MobileNet_3Layers was selected as the final model due to its high accuracy and f1 score. More details are in the [CapStone0915.ipynb](https://github.com/mengfei-liu/Capstone/blob/master/CapStone0915.ipynb) Jupyter Notebook.

## Findings and Conclusion:

Based on the confusion matrix below that the model predicted on the test set, we found that our model has a pretty good capability of identifying the categories such as ‘talking_phone’, ‘texting_phone’ and ‘turning’, however, ‘other_activities’ has the most misclassifications. The reason for this is probably some motion in the other_activities category, like holding a water bottle, is similar to holding a phone.

| ![image](https://github.com/mengfei-liu/Capstone/blob/master/img/3.png) | ![image](https://github.com/mengfei-liu/Capstone/blob/master/img/4.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image](https://github.com/mengfei-liu/Capstone/blob/master/img/5.png) | ![image](https://github.com/mengfei-liu/Capstone/blob/master/img/6.png) |
| ![image](https://github.com/mengfei-liu/Capstone/blob/master/img/7.png) | ![image](https://github.com/mengfei-liu/Capstone/blob/master/img/8.png) |

![image](https://github.com/mengfei-liu/Capstone/blob/master/img/2.png)

**Video Analyze**

```
[![Predicted by the model](https://img.youtube.com/vi/JM9rxjO0xyg/0.jpg)]
(https://www.youtube.com/watch?v=JM9rxjO0xyg)
```

Through our process of building models, we also learned that adding layers increases the number of weights in the network and the model complexity. Without a large training set, an increasingly large network is likely to overfit and in turn reduce accuracy on the test data. There are other ways of increasing the accuracy of a network of existing depth, like dropout, changing model hyperparameters, image augmentation, or even reducing the model complexity. Hence, increasing layers is not the first option for increasing accuracy. Overall, a well trained convolutional neural network model is able to detect distracted driving behaviors with a limited resolution camera and provide feedback in real time in a proactive manner.
In terms of business implementation, it is not difficult to put the model inside an embedded system algorithm, other hardware need for our system are a relatively cheap camera and a buzzer. For the future version, more parameters tuning needs to be done. We also need more dataset to train our model, make the model more generalization, in order to deal with different angles of the picture, different colors, etc.