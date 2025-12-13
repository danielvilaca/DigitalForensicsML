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
aditional_stopwords = ["0f", "attached", "cc", "com", "corp", "doc", "ect", "ees", "email",
                       "enron","enronxgate", "forwarded","hou","mail","message","na","new","original","pm",
                       "recipient", "said","sent","subject","year","2000","2001"]

smallnum_stopwords = [str(i).zfill(2) for i in range(100)]
stop_words = stop_words.union(smallnum_stopwords)
stop_words = stop_words.union(aditional_stopwords)
cvect = CountVectorizer(stop_words=list(stop_words))

X_train_matrix = cvect.fit_transform(df['message'].astype(str))
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(X_train_matrix)
print("Clustering Completed!")

#extracting top 20 words per cluster
centre = kmeans.cluster_centers_
words = cvect.get_feature_names_out()

for i, center in enumerate(centre):
    top_words_indices = np.argsort(center)[::-1][:20]
    top_words = [words[ind] for ind in top_words_indices]
    print(f"Top 20 Words: Cluster {i}: {top_words}")

#list cluster counts
clusters = kmeans.labels_
counts = pd.Series(clusters).value_counts().sort_index()
print(counts)

#test email text
test_email = "Make sure you check the folder before you tranfer it to me. Also give me the data details like filename, version, origin, date, etc."
test_email_transformed = cvect.transform([test_email])
predicted_cluster = kmeans.predict(test_email_transformed)
print(f"The test Email fits into Cluster {predicted_cluster[0]}")
