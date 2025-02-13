# ml-lab-cse-5211
# Project Name: Sentiment Analysis on Movie Reviews using Machine Learning Techniques
Supervised By:<br>
Md. Mynoddin <br>
Assistant Professor<br>
Department of  CSE

Submitted By:<br>
Team Name: Nexus<br>
Sagar Mohajan (2018-15-01)<br>
Momotaj jahan moon  (2018-15-42)

# Overview 
An entertainment industry that is always evolving, the public's perception of a movie often determines its chances of success or failure. These days, a lot of information on how people react to movies comes from reviews, comments, and opinions from the public. Analyzing these emotions may assist in identifying the elements of a successful movie and act as a guide for producers and marketers when making decisions. Comments and opinions in movie reviews are a great way to get a sense of how the general audience feels. This project explores the field of sentiment analysis with an emphasis on assessing the thoughts and emotions stated in movie reviews. 
The aim of this project is to utilize cutting-edge machine learning methods to examine and categorize the emotions found in movie reviews. This entails forecasting whether a review will be favorable, unfavorable, or neutral, offering insightful information about how the audience will respond. Five different machine learning approaches are used in this study: Random Forest Classifier, Multinomial Naive Bayes, XGB Classifier, Linear Support Vector Classifier (SVC), and Logistic Regression. Every technique contributes a unique perspective to sentiment analysis, helping to provide a thorough grasp of audience responses. Through the application of these methodologies to the analysis of feelings found in movie reviews, the research hopes to offer insightful information to producers, directors, and other entertainment industry players. Making educated judgments about movie production, marketing, and general audience interaction requires having a thorough understanding of audience feelings. Finally, analyzing sentiment in movie reviews with state-of-the-art machine learning techniques offers a deeper comprehension of audience reactions. The goal of this research is to provide significant insights to the film business, enabling better decision-making and developing a closer relationship between filmmakers and their audience. Understanding audience emotions becomes a crucial skill for anyone managing the complex business of film production and distribution as the cinematic environment changes more. 

# Problem Statement 
Movies are a big deal, and what people say about them matters a lot. Anyone may use the internet to evaluate movies and share their opinions with others. The problem, though, is that there are a ton of reviews online, making it difficult for filmmakers to know what people actually think. Furthermore, as people express their emotions in different ways, some evaluations might be challenging. Additionally, in order to gain a deeper understanding of these reviews, we must select the appropriate machine learning techniques. It's also true that various people have diverse tastes in movies, which comes in a variety of forms. Therefore, it can be challenging to gauge 
how different types of movies affect different people. 

# Motivation  
This project uses five different sentiment analysis approaches on movie reviews to address the changing dynamics of audience sentiment in the digital era. Given the significant influence of user-generated information on the formation of cinematic perceptions, it is critical to comprehend the complex emotions conveyed in reviews. The study seeks to offer insightful information about the emotional impact of films to reviewers, directors, and the film industry as a whole. We want to determine which approaches—machine learning, natural language processing, deep learning, lexicon-based methods, and ensemble techniques—are most useful for identifying sentiments. This work advances our understanding of the complex interaction between audience emotions and cinematic experiences, which informs future developments in this vital nexus of entertainment and technology. It also makes a contribution to the field of 
sentiment analysis.

# Objective of this Research 
The primary goal of this study is to develop an effective model for sentiment analysis based on movie review. In order to achieve this goal, the other goals are: <br>
1 To predict the type of a movie upon reading the review of that movie. <br>
2 To determine the degree to which the public views the film based on reviews.<br> 
3 To evaluate the mood in relation to the audience's reaction. <br>
4 To enhance the marketing initiatives.

# Methodology 
I gathered data from Kaggle. Then used regular expressions to clean up our data. In the preprocessing stage, all types of numbers, punctuation, and stop-words are eliminated. The data had already been labeled. I utilized TF-IDF with the bigram feature for feature extraction after cleaning the dataset. Next, feed the models with the TF-IDF features. I initially divided the dataset into 70% and 30% in order to train the models. 70% of the dataset was utilized to train the models and utilized 30% of the dataset to test the models. I employed five well-known machine learning methods in the models. These include XGBoost, Random Forest, Logistic Regression, Multinomial Naïve Bayes, and linear SVC. The prediction portion of our task comes next. For this section, i create an unlabeled validation dataset with a random movie review. I employed my optimal model for the purpose of classifying our validation dataset. During this classification procedure, my unlabeled data were divided into two groups: positive and negative. I measure my classification result and convert it to a percentage after the classification procedure is complete. Then compare percentage; if it is higher for the positive than for the negative, my computer anticipates that this movie review will be positive. The movie could be negative if there are more negative comments than good ones. 
![image](https://github.com/user-attachments/assets/14755cd4-af59-4884-a1ca-4640512f5454)

# Data Collection 
The dataset was obtained from the well-known data science and machine learning resource site Kaggle. The dataset was chosen to ensure it included the characteristics and information needed for analysis, considering its relevance to my study subject. For my project, Kaggle offers a wide variety of community-contributed datasets, giving me access to a high-quality and thoroughly described dataset. Through exploratory data analysis and background investigation, the data's quality and dependability were confirmed once it was downloaded utilizing Kaggle's userfriendly interface. This dataset provides a solid basis for my study by enabling in-depth examination and significant insights into the selected topic. In this dataset there are 50000 reviews of movie. Out of 50000 reviews, 25000 reviews are positive and rest of the 25000 are negative sentimental review. 

#  Data Pre-Processing 
After collecting data, i have to preprocess this dataset. For preprocessing part, i used: <br>
1 <b>Removing Punctuations:</b> Now, remove all punctuation. Which in the categorization portion is completely pointless. Thus, I create a dataset without of 
  punctuation. <br>
2 <b>Removing digit:</b> Remove All Digits Like: 0,1,2,3,4,5,6,7,8,9 <br>
3 <b>Removing Stopwords:</b> Stopwords mainly meaningless words. So, i removed all Stopwords. <br>
That‘s are: <br>
[ ‗i‘,‘ ‗me‘, ‗my‘, ‗myself‘, ‗we‘, ‗our‘, ‗ours‘, ‗you‘, ‗your‘, ‗he‘, ‗him‘, ‗his‘, ‗it‘, <br>
‗its‘, ‗who‘, ‗which‘ ‗is‘, ‗are‘, ‗was‘, ‗be‘, ‗have‘, ‗has‘, ‗do‘, ‗a‘, ‗an‘, ‗the‘……]  <br>
4 <b>Stemming:</b> The technique of stemming involves returning words with variation to their basic forms. I stem my dataset utilizing the stemming procedure, which turns every remark included in the dataset into a root word. displaying a sample of my dataset's stemming effects:<br>

![image](https://github.com/user-attachments/assets/f84fc3cf-9d47-4389-ac8f-b55bd0fa8876)

5 <b>Lemmatization :</b> Lemmatization is a linguistic process that unifies word variants by reducing words to their basic or root form, or lemma. Lemmatization takes the 
word's context and grammatical structure into account, in contrast to stemming. 

![image](https://github.com/user-attachments/assets/f9fbaac0-1f78-4f46-a8c2-ed8185caa4f0)

6 <b>Feature-Extraction</b> 
Every text is represented in feature extraction as a collection of features known as a feature vector. Three distinct feature categories are examined for in the proposed model. 
1. Unigram with TF-IDF.  
2. Bigram with TF-IDF 
3. Trigram with TF-IDF

7 <b>TF-IDF Vectorizer:</b> 
Term Frequency - Inverse Document Frequency is referred to as TF-IDF. Term frequency is the result of dividing the total number of words in a text by the number of 
terms that occur in it. IDF is used to reduce the size of words that often occur in manuscripts. The method's mathematical formula is as follows:

![image](https://github.com/user-attachments/assets/075151ad-2fb1-40c4-a02c-1f14e30a6a24)

# Classification Model 
An overview of the algorithms i frequently used for my thesis work is given in this section. Then often test five machine learning algorithms on my system. These five 
algorithms are LR, MNB, XGB, RF, and Linear SVC.
1.  <b>Logistic-Regression</b>  
A supervised learning classification approach called logistic regression is used to estimate the chance of a target variable. Due to the bidirectional structure of the 
dependent variable, there can only be two potential classes. Therefore, don't be confused by its name. This isn't a regression procedure; it's a categorization. It is employed to estimate discrete values based on a given collection of independent variables, such as binary values like 0/1, yes/no, and true/false.  It uses logistic function fitting to estimate the likelihood of an event occurring. It is thus sometimes referred to as logistic regression. Because it forecasts the likelihood, the range of its output values is 0 to 1. Take a look at the image below for a thorough understanding of logistic regression.
![image](https://github.com/user-attachments/assets/35c2a3c6-076f-4459-ad03-b152076036c9)

P(Y=1) is mathematically predicted by a logistic regression model as a function of X. One of the most basic machine learning algorithms, it may be applied to a number of categorization issues, including diabetes prediction and spam identification.

2.  <b>Random Forest Classifier </b>
Popular machine learning method Random Forest is a member of the supervised learning approach. It may be applied to ML issues involving both classification and regression. Its foundation is the idea of ensemble learning, which is the act of merging several classifiers to solve a challenging issue and enhance the model's functionality. According to its name, "Random Forest could be a classifier that contains variety of call trees on numerous subsets of the given 5 dataset and takes the common to boost the prophetic accuracy of that dataset." The random forest forecasts the final result by taking the predictions from each call tree and supporting them with 
the majority of votes, as opposed to relying just on one call tree. The Random Forest method is explained in the image below:

![image](https://github.com/user-attachments/assets/96e2728d-ceb1-4c94-91d7-7d8876708255)

3. <b>Multinomial-Naive-Bayes </b>
A probabilistic learning technique that is primarily utilized in natural language processing (NLP) is the Multinomial Naive Bayes algorithm. The method guesses the tag of a text, such as an email or newspaper article, and is based on the Bayes theorem. It determines each tag's likelihood for a given sample and outputs the tag with the highest probability. The Naive Bayes classifier is a group of many algorithms that are united by the idea that each feature being categorized is independent of every other feature. The other feature's presence or absence is unaffected by the existence of one feature. When dealing with situations involving numerous classes and text data processing, Naive Bayes is a potent method. Since the Naive Bayes theorem depends on the notion of the Bayes theorem, it is crucial to comprehend it before attempting to comprehend how the latter works. Thomas Bayes developed the Bayes theorem, which determines the likelihood of an event happening based on knowledge of its relevant circumstances in the 
past. It is predicated on the subsequent equation:
p(A|B) = p(A) * p(B|A)/p(B)
Where the probability of class A is being calculated in the case where predictor B is already available. 
P (A) = prior probability of class A  
P (B) = prior probability of B  
P (B|A) = occurrence of predictor B given class A probability

4. <b>Linear SVC </b>
For classification tasks, a machine learning approach called Linear Support Vector Classifier (Linear SVC) is employed. It functions by determining which hyperplane in the 
feature space best divides various classes. When there is a roughly linear connection between characteristics and classes, linear SVC performs especially well. The algorithm's goal during training is to maximize the margin between classes, which is a measure of how far off each class's closest data points are from the hyperplane. Take a look at the image below for a thorough understanding of Linear SVC:

![image](https://github.com/user-attachments/assets/b85bbc0d-95cf-4e2a-8c7b-dbcd8d4c786b)

High-dimensional datasets can benefit from linear SVC's ability to handle massive volumes of data effectively. Since it is a linear model, characteristics and classes are assumed to have a linear relationship. In spite of its ease of use, Linear SVC frequently exhibits strong performance in real-world scenarios and is extensively employed in diverse fields including bioinformatics, picture recognition, and text categorization. Linear SVC is an adaptable technique in the field of linear classification because regularization parameters may be adjusted to regulate the trade-off between attaining a large margin and reducing classification mistakes.<br>

5. <b>XGBoost </b>
The decision-tree based ensemble machine learning technique XGBoost uses a gradient boosting architecture. In prediction problems involving unstructured data, artificial neural networks often perform better than any other algorithms or frameworks (images, text, etc.).
For small- to medium-sized structured/tabular data, decision tree-based algorithms are now regarded as best-in-class. Please see the graph below for an overview of the 
evolution of tree-based algorithms over time. XGBoost is a popular gradient boosting program. Let's discuss some of the features of XGBoost that make it so fascinating. 
   - Regularization: XGBoost may use both L1 and L2 regularization to punish complex models. Regularization can help prevent overfitting.  
   - Handling sparse data: Missing values or data processing techniques like one-hot encoding cause data to become sparse. A sparsity-aware split discovery approach is used 
by XGBoost to address different types of sparsity patterns in the data.  
   - Weighted quantile sketch: Most existing tree-based algorithms can find the split sites (using quantile sketch technique) when the data points have equal weights. Still, they are unable to handle weighted data. 
   - Block structure for parallel learning: This is made feasible by the system's block structure. Data is stored in memory using blocks that are classified.  
   - Cache awareness: Non-contiguous memory access is required for XGBoost to acquire the gradient statistics per row index. As a result, XGBoost was developed to fully use the hardware. Each thread allots internal buffers to store the gradient statistics in order to do this.  
   - Out-of-core computing: This feature maximizes the utilization of the available disk space and increases its efficiency while working with huge datasets that cannot fit in memory.

# Required Tools and Technologies 
# Programming Language
1.  <b>Python</b>
   For all of my work, i using Python 3.7 as our programming language. It's an amazing language. Python has many resources and has the potential to be a very strong artificial language. Its extensive application in AI, deep learning, machine learning, and knowledge science. 
 -  Pandas: Pandas is a Python language library function.  Pandas is the preferred tool for manipulating and analyzing data. Wes McKinney created the sophisticated information manipulation tool known as Pandas. Its fundamental structure is known as the Data Frame, and it is based on the NumPy library. With rows of observations and columns of variables, the Data Frame facilitates the manipulation and archiving of tabular data.
 -   Sklearn: An estimator for classification in Scikit-learn may be a Python object that uses the fit(X, y) and predict (T) methods. An exemplar of an estimator is the 
sklearn.svm.SVC category. 
 - Seaborn: Based on matplotlib, Seaborn is a Python data visualization package. It offers a sophisticated drawing tool for creating eye-catching and educational statistics visuals.
 - NLTK: A Python library that may be used for NLP is called Natural Language Toolkit, or NLTK. A significant portion of the data you may be examining is unstructured and 
includes text that is understandable by humans. You must preprocess the data before you can programmatically evaluate it.

# Platform 
# Google Colab 
In order to do this study, i use Google Colab. An online platform called Google Colab uses the artificial language Python. Every Python module and resource available on Google Colab is available for purchase. Its own on-board memory, graphics card, and RAM allow users to navigate around easily. 
Perhaps Google Colab is a simple platform.

# Result analysis and Model evaluation 
The findings and a discussion of the system's accuracy in various algorithm and feature strategies for the machine learning-based sentiment analysis of movie reviews are presented in this chapter. This section also covers the precision, recall, F1 score, ROC curve, and accuracy of sentiment prediction using binary category using various algorithms.
#  Result Comparison 
Two of the models—Logistic Regression and Linear Support Vector Classifier (Linear SVC)—performed exceptionally well in the extensive investigation of sentiment analysis on movie reviews using machine learning techniques, outperforming the accuracy of the other three models. The well-known straightforward and efficient method of modeling linear connections, logistic regression, was remarkably good at capturing the subtle patterns present in the sentiment data from movies. The dataset's underlying structures appeared to be in line with the linearity requirements of logistic regression, which helped the model achieve an accuracy of more than 88%. In a similar vein, the linear classifier Linear SVC performed admirably, particularly in situations where the data may be divided by a hyperplane. It was shown that Linear SVC worked well for correctly categorizing movie reviews in the context of sentiment analysis, where feelings are frequently seen within linear bounds. Its strong performance—which also exceeded 88% accuracy—showcased how well its linear classification strategy suited the current problem. On the other hand, the accuracy of the remaining three models (XGBoost, Random Forest, and Naive Bayes) was just marginally higher, ranging from 80 to 85%. These models could have faced difficulties because of the subtleties of film sentiment, which are sometimes characterized by complex non-linear interactions. Given its presumption of feature independence, Naive Bayes may not have been able to accurately identify complicated connections in the data. Although they are strong ensemble techniques, Random Forest and XGBoost might not have been as appropriate for the particular patterns seen in the sentiment dynamics of movie reviews. The models' differing levels of success highlight how crucial it is to choose models that are compatible with the complexity of the data being examined. 

# The confusion matrix and ROC curve 
To select the optimal model for my task, I make use of the ROC curve. An assessment measure for binary classification issues is the Receiver Operator Characteristic (ROC) curve. I determine the roc curve score for each method utilized in that experiment using binary classification. I used TF-IDF for feature extraction for all the models. 

![image](https://github.com/user-attachments/assets/3246e5af-db0c-421f-9a35-542ee308add5)

# Confusion Matrix: 
In machine learning, a tabular representation that summarizes a classification model's performance is called a confusion matrix. There are four types of entries in 
it: false positives, which are wrongly anticipated positive occurrences, false negatives, which are incorrectly projected negative instances, and true positives, which are accurately predicted positive instances. The model's recall, accuracy, precision, and F1 score are all assessed using these criteria. Recall gauges the model's capacity to capture all positive occurrences, precision evaluates the model's ability to prevent false positives, and accuracy is the ratio of right forecasts to total predictions. The F1 score balances recall and accuracy by combining them into a single rating. Confusion matrices are very helpful in determining areas for improvement in classification tasks and in gauging a model's performance on particular classes.

![image](https://github.com/user-attachments/assets/88ab6bd9-bc35-4460-9267-4eae197eae6e)
<br> Figure : Confusion Matrix of Multinomial Naive Bayes 

![image](https://github.com/user-attachments/assets/f7a62f4b-7410-458b-afbb-b22f3ccfc3ad)
<br> Figure : Confusion Matrix of Random Forest Classifier

![image](https://github.com/user-attachments/assets/74297643-8e88-4a78-b145-54eabd15de19)
<br> Figure : Confusion Matrix of Logistic Regression

![image](https://github.com/user-attachments/assets/4c2c2b35-505f-4055-a7b7-406e826ad6d1)
<br> Figure : Confusion Matrix of Linear SVC

![image](https://github.com/user-attachments/assets/93690b06-f2e6-4be5-8a3d-441bf3d66145)
<br> Figure : Confusion Matrix of XGBoost

# ROC-curve: 
In machine learning, a graphical depiction known as a Receiver Operating Characteristic (ROC) curve is used to assess how well binary classification models perform. It 
shows how, at various categorization thresholds, the true positive rate (sensitivity) and false positive rate (specificity) are traded off. Plotting these rates at different threshold levels, the curve illustrates how well the model can differentiate between positive and negative occurrences. A ROC curve reaching the upper-left corner, which denotes great sensitivity and a low false positive rate, would be indicative of an ideal model. A greater Area Under the Curve (AUC) indicates better model discrimination. The AUC measures the total performance. Because it offers a thorough understanding of a model's discriminating capacity regardless of the decision threshold selected, the ROC curve is especially helpful in situations when the class distribution is unbalanced. It helps choose the best threshold for a certain application by taking into account the intended ratio of specificity to sensitivity. 

![image](https://github.com/user-attachments/assets/9caabd97-4d77-4365-8fc8-2b09decbf9b2)
<br> Figure : ROC curve of Multinomial Naive Bayes 

![image](https://github.com/user-attachments/assets/0b237ba7-b02a-4687-adc1-3e780475c6d3)
<br> Figure : ROC curve of Random Forest Classifier     

![image](https://github.com/user-attachments/assets/7425e5f4-3ddf-45a4-a91b-46b770db64e3)
<br> Figure : ROC curve of Logistic Regression 

![image](https://github.com/user-attachments/assets/e122f619-cd78-4d74-8818-6107ad2744db)
<br> Figure : ROC curve of Linear SVC 

![image](https://github.com/user-attachments/assets/96f05b94-ce79-4658-b927-b8210ee80bba)
<br> Figure : ROC curve of XGBoost  

# Conclusion 
To sum up, this thesis used a variety of machine learning models to explore the field of sentiment analysis on movie reviews. With an accuracy rate of more than 88%, Logistic Regression and Linear Support Vector Classifier stood out as the best algorithms. Their accomplishments highlight how important linear classification techniques are for capturing the nuanced details of cinematic emotion. The results imply that these models' linearity assumptions were appropriate for the particular patterns found in the sentiment dynamics of various movie reviews. Although XGBoost, Random Forest, and Naive Bayes all had somewhat worse results, with accuracies of 80–85%, the study emphasizes how crucial it is to choose a model that is appropriate for the dataset's complexity. In summary, this study makes a significant contribution to the area of sentiment analysis by offering a sophisticated knowledge of the effectiveness of machine learning algorithms in interpreting audience feelings in the dynamic entertainment business.     

#  Future Work 
The work creates opportunities for further investigation into the field of machine learning-based sentiment analysis of movie reviews. Given the dynamic nature of language and sentiment expressions, more research may entail improving the efficiency and accuracy of current models. A more thorough grasp of audience responses may be obtained by examining the effects of further factors, such as user demographics or the sentiment of reviews over time. Furthermore, using deep learning techniques like neural networks could reveal subtle patterns in the data that conventional machine learning models might miss. Another interesting approach is the incorporation of sentiment analysis into recommendation systems for tailored content distribution. Future research may examine sentiment analysis across multimedia content types as the entertainment industry develops, extending the use of machine learning to comprehend audience sentiments in various language and cultural situations.  

# References
[1] Baid, P., Gupta, A., & Chaplot, N. (2017, December 15). Sentiment Analysis of Movie Reviews using Machine Learning Techniques. International Journal of Computer Applications, 179(7), 45–49.  
[2] Sharma, H., Pangaonkar, S., Gunjan, R., & Rokade, P. (2023, June 1). Sentimental Analysis of Movie Reviews Using Machine Learning | ITM Web of Conferences. Sentimental Analysis of Movie Reviews Using Machine Learning | ITM Web of Conferences.  
[3] Kari, Hemanth Kumar. "Sentimental Analysis of Movie Tweet Reviews Using Machine Learning Algorithms." (2024). Proceedings of the 57th Hawaii International Conference on 
System Sciences.<br>
[4] Dashtipour, K., Gogate, M., Adeel, A., Larijani, H., & Hussain, A. (2021, May 12). Sentiment Analysis of Persian Movie Reviews Using Deep Learning. MDPI.  
[5] R. R. Chowdhury, M. Shahadat Hossain, S. Hossain and K. Andersson, "Analyzing Sentiment of Movie Reviews in Bangla by Applying Machine Learning Techniques," 2019 
International Conference on Bangla Speech and Language Processing (ICBSLP), Sylhet, Bangladesh, 2019, pp. 1-6, 
