import re
import numpy as np
import pandas as pd
import nltk
dataset=pd.read_csv('C:\\Users\\aksha\\Documents\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\Natural_Language_Processing\\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
list=[]
nltk.download('stopwords')

for i in range(0,1000):

    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    ps=PorterStemmer()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    list.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(list).toarray()
y=dataset.iloc[:,1].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
