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
