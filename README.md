# ml-lab-cse-5211
# Project Name: Sentiment Analysis on Movie Reviews using Machine Learning Techniques
Supervised By:
Md. Mynoddin
Assistant Professor
Department of  CSE

Submitted By:
Team Name: Nexus
Sagar Mohajan (2018-15-01)
Momotaj jahan moon  (2018-15-42)

# Overview 
An entertainment industry that is always evolving, the public's perception of a movie often determines its chances of success or failure. These days, a lot of information on how people react to movies comes from reviews, comments, and opinions from the public. Analyzing these emotions may assist in identifying the elements of a successful movie and act as a guide for producers and marketers when making decisions. Comments and opinions in movie reviews are a great way to get a sense of how the general audience feels. This project explores the field of sentiment analysis with an emphasis on assessing the thoughts and emotions stated in movie reviews. 
The aim of this project is to utilize cutting-edge machine learning methods to examine and categorize the emotions found in movie reviews. This entails forecasting whether a review will be favorable, unfavorable, or neutral, offering insightful information about how the audience will respond. Five different machine learning approaches are used in this study: Random Forest Classifier, Multinomial Naive Bayes, XGB Classifier, Linear Support Vector Classifier (SVC), and Logistic Regression. Every technique contributes a unique perspective to sentiment analysis, helping to provide a thorough grasp of audience responses. Through the application of these methodologies to the analysis of feelings found in movie reviews, the research hopes to offer insightful information to producers, directors, and other entertainment industry players. Making educated judgments about movie production, marketing, and general audience interaction requires having a thorough understanding of audience feelings. Finally, analyzing sentiment in movie reviews with state-of-the-art machine learning techniques offers a deeper comprehension of audience reactions. The goal of this research is to provide significant insights to the film business, enabling better decision-making and developing a closer relationship between filmmakers and their audience. Understanding audience emotions becomes a crucial skill for anyone managing the complex business of film production and distribution as the cinematic environment changes more. 

# Problem Statement 
Movies are a big deal, and what people say about them matters a lot. Anyone may use the internet to evaluate movies and share their opinions with others. The problem, though, is that there are a ton of reviews online, making it difficult for filmmakers to know what people actually think. Furthermore, as people express their emotions in different ways, some evaluations might be challenging. Additionally, in order to gain a deeper understanding of these reviews, we must select the appropriate machine learning techniques. It's also true that various people have diverse tastes in movies, which comes in a variety of forms. Therefore, it can be challenging to gauge 
how different types of movies affect different people. 

# Objective of this Research 
The primary goal of this study is to develop an effective model for sentiment analysis based on movie review. In order to achieve this goal, the other goals are: 
1 To predict the type of a movie upon reading the review of that movie. 
2 To determine the degree to which the public views the film based on reviews. 
3 To evaluate the mood in relation to the audience's reaction. 
4 To enhance the marketing initiatives.

# Methodology 
I gathered data from Kaggle. Then used regular expressions to clean up our data. In the preprocessing stage, all types of numbers, punctuation, and stop-words are eliminated. The data had already been labeled. I utilized TF-IDF with the bigram feature for feature extraction after cleaning the dataset. Next, feed the models with the TF-IDF features. I initially divided the dataset into 70% and 30% in order to train the models. 70% of the dataset was utilized to train the models and utilized 30% of the dataset to test the models. I employed five well-known machine learning methods in the models. These include XGBoost, Random Forest, Logistic Regression, Multinomial Naïve Bayes, and linear SVC. The prediction portion of our task comes next. For this section, i create an unlabeled validation dataset with a random movie review. I employed my optimal model for the purpose of classifying our validation dataset. During this classification procedure, my unlabeled data were divided into two groups: positive and negative. I measure my classification result and convert it to a percentage after the classification procedure is complete. Then compare percentage; if it is higher for the positive than for the negative, my computer anticipates that this movie review will be positive. The movie could be negative if there are more negative comments than good ones. 
![image](https://github.com/user-attachments/assets/14755cd4-af59-4884-a1ca-4640512f5454)
