# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:33:10 2018

@author: Gourav Sharan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

#Data Cleaning
import re

import nltk # Step 3
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #Step 4
ps = PorterStemmer()
corpus = []
for i in range(0,1000): #Step 6
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) # Step 1
    review = review.lower() #Step 2
    review =review.split() #Converts the string into a list
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) # Step 5
    corpus.append(review)
               
#Creating the Bag of Words model Step 7
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Create classification model using naive bayes

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
precision = cm[1,1]/(cm[1,1]+cm[0,1])
recall = cm[1,1]/(cm[1,1]+cm[1,0])
f1_score = 2 * precision * recall/(precision + recall)



# Step 1: Eliminate all the unwanted characters from the review
""" The sub method in the re library is used to keep only the letter from A-Z.
The first parameter is used to tell what to be replaced or what not to be replaced. It starts with the leftmost 
character.The "^" is used to tell that the letters are not to be replaced and rest to be replaced. 
The second parameter tells us the replacement character. Here the replacement is an empty space specified by ' '.
The 3rd parameter is the input string for which the replacement is to be carried out.
"""

#Step 2
""" Convert all the characters in lower case"""

#Step 3
""" We now need to remove the irrelevant words like articles, prepositions that are not useful for ML models
for predicting the review. Hence we import the library named "nltk", from which we download the "stopwords" 
list that contains all the irrelevant words. Then we have to import the stopwords function from the nltk library
into our program in Spyder.
The review is converted from the str type into list of words using the split() function
Now a for loop is run to check and compare the review list and the stopwords list. if not is used so that,
any matching word will not be added to our review.
The set() function is mostly useful in large set of words or articles. As this function entirely makes the 
words as as input. The processing is much faster for the program in the form of a set than in a list.
"""

#Step 4
""" Stemming process. This is finding the root word of the words in our review list. 
PorterStemmer is the class imported from the nltk library. 
ps is the object of the class and the word as soon as filtered out for loop, gets stemmed and is added to the 
review list.
"""

#Step 5
"""join back the list into string"""

#Step 6
""" Repeat the same process for all the reviews"""

#Step 7
""" Create the bag of words model. EEach word gets its own column and is given a value 1 if it appears in the 
rows and given 0 if it is not present in that row.
Now the problem is converted into classification problem with a sparse matrix
"""
