import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.cluster import KMeans


#importing email dataset
df = pd.read_csv('./emails.csv')
print(df.shape)
df = df.dropna()
print(df.shape)

#extracting features to train the model
stop_words = _stop_words.ENGLISH_STOP_WORDS
cvect = CountVectorizer(stop_words=list(stop_words))
X_train_matrix = cvect.fit_transform(df['message'].astype(str))
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(X_train_matrix)
print("Clustering Completed!")
