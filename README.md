# Capstone
BrainStation Capstone Project
In this project, I structured a CNN framework using machine learning and transfer learning techniques to process image features. 
The object of this project is to build a machine learning model that can recognize 5 different behavior categories. 
When the distracted driving is happening, the model is able to remind the driver to prevent traffic accidents.

Data Source :
The dataset comes from the Kaggle called ‘Driver Behavior Dataset’.
The dataset consists of 5 categories:
  ● saft_driving
  ● talking_phone
  ● texting_phone
  ● turning
  ● other_activities
There are 10766 images in the format of JPG and PNG all across the 5 categories.

Data Preprocessing & EDA:
  ● Filter out the corrupted files: to prevent null values
  ● Check data distribution: to prevent class imbalance(class imbalance will make model more capable of recognizing the class with large portion of data)
  ● Resized all the images: tensorflow model requires all the images being the same size
  ● Reconstruct the data set: split it into train set and test set, using test set to provide a totally unbiased estimate of the model's performance. This is data the model has never seen; it should serve as a good predictor for the model's performance once deployed and making prediction on new data.

After cleaning and reconstruction, we have a balanced data set without any null value:
