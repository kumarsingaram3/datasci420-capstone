# datasci420-capstone

Capstone - Classifying defective products in the diaper manufacturing process. This project is a part of the Data Science 420 capstone project at the University of Washington. You can find detail about the diaper manufacturing process [here](https://github.com/kumarsingaram3/datasci420-capstone/blob/main/Diaper%20Manufacturing%20Process.docx). To read or run the script for this project, click [here](https://github.com/kumarsingaram3/datasci420-capstone/blob/main/DataSci420-Capstone.ipynb).

## Table of Contents

- [Project Description](#project-description)
- [Findings](#findings)
- [Technologies](#technologies)
- [Methods](#methods)
- [Installation](#installation)

## Project Description

To ensure or predict quality, a diaper manufacturer needs to monitor every step of the manufacturing process with sensors such as heat sensors, glue sensors, glue level, etc. To classify defective products in the diaper manufacturing process, we used the SECOM data set from the UCI Machine Learning Repository, which contained these sensors. This data set was then joined to a data set of labels, indicating whether a product was defective or not. 

The goal of this project was to go through a real business case, clean & prepare data, handle class imbalance, and build & test each of the algorithms learned in class.

## Findings

In this project, we were given approximately 1,600 samples of data on 590 features, which we did not have the names for. Each of these features are sensors in the manufacturing process to help us detect when there is a faulty product. You can find a data flow diagram that picturizes how we narrowed our feature set down and balanced target classes:

![DFD_capstone](https://user-images.githubusercontent.com/75543007/106971888-cb73ab80-671d-11eb-8a27-66ed59e0abb3.PNG)

As the diagram shows, we had to deal with about 7% of products being faulty, which means we had a very imbalanced data set. To solve this problem, we used a method called SMOTE, which helped us balance the classes 50-50, giving our models an equal number of samples to work with for both classes.

To narrow down the feature set, we used two methods called Mutual Information Classification and Recursive Feature Elimination (RFE) respectively. The first method is an efficient way to help understand the information an individual independent feature provides us about the dependent variable. Using this, we narrowed the feature set from 590 to 125 features. Then, we used RFE to iteratively eliminate features it did not find to be important by ranking all the features according to importance. In this process, RFE only removed 3 features, leaving us with 122 out of the 125 total features.

![RFE](https://user-images.githubusercontent.com/75543007/106972615-35d91b80-671f-11eb-91ba-0d95542a5331.PNG)

In the modeling process, we decided to test 5 different models. We started by testing a Decision Tree, since it would provide interpretable & potentially important insights:

![dtree_capstone](https://user-images.githubusercontent.com/75543007/106972807-9e27fd00-671f-11eb-8a7e-8ad399cd09fc.PNG)

The performance metrics we wanted to optimize were Precision, Recall, and Area Under the Curve. Because we had an imbalanced data set, it was very important to find a model that would do a good job of identifying the positive cases (defective products). After looking at a decision tree, we tested a Random Forest, SVM, a Fully Connected MLP Classifier, and finally a Neural Network using TensorFlow with Dropout. 

We then tuned the TensorFlow model until reaching convergence between the training/validation sets and validation loss began to plateau.

![TF_model_loss](https://user-images.githubusercontent.com/75543007/106973169-55247880-6720-11eb-9245-2e09bfca7e15.PNG)

After tuning the model, we tested the TensorFlow model against the Random Forest model since it was the best performing model on our validation set. And ultimately, we finalized the TensorFlow model, as its performance was significantly higher than that of the Random Forest, achieving a 72% AUC vs only a 53% AUC for the Random Forest. 

The takeaways from this model, in terms of its potential impact on the business needed to be considered cautiously though. With 1567 observations and only ~7% of them, as defective; the models built here were not able to produce very consistent results on unseen data. To optimize these models, we suggested collecting more data (including non sensor data as well) to be used in training. This would help the models built here to learn the underlying structures and relationships within the data better. As expected, each of the models predicted the majority class very well, but could not identify positive cases well.

If false positives/negatives are expected to be very costly to the organization and there are other investment risks associated with building processes around an inconsistent model such as this, it may be more valuable to build a rule based classifier using the most predictive sensors from our analysis and other related domain knowledge. Doing this could be both interpretable for decision makers and also moderately accurate in production, all without having to spend the resources required to deploy a fully connected neural network.

## Technologies

* Python
* Pandas
* Numpy
* Jupyter
* Sklearn
* TensorFlow

## Methods

* Feature Selection
* Handling Class Imbalance
* Machine Learning Techniques

## Installation

* Clone this repo to your computer
* Save [SECOM data](https://github.com/kumarsingaram3/datasci420-capstone/blob/main/secom.data) to your computer (into 'data' folder)
* Save [labels data](https://github.com/kumarsingaram3/datasci420-capstone/blob/main/secom_labels.data) to your computer (into 'data' folder)



