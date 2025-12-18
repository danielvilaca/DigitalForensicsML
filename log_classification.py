import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer #better than CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from time import perf_counter
from sklearn.metrics import accuracy_score, confusion_matrix

from urllib.parse import urlparse, parse_qs
import re


def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'alphabets_count': sum(c.isalpha() for c in url),
        'special_characters_count': sum(c in ['-', '_', '@', '/', '?', '=', '.', '%', '&', '!', ';', '+', ',', '*', '~', '|'] for c in url),
        'digits_count': sum(c.isdigit() for c in url),
        'has_https': 1 if "https" in url else 0,
        'has_ip': 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0,
        'subdomain_count': url.count('.'),
        'parameters_count': len(parse_qs(parsed_url.query)),
        'suspicious_tlds': 1 if any(tld in url for tld in ['.zip', '.rar', '.exe', '.dmg', '.sh', '.bin', '.xyz', '.top', '.pw', '.icu', '.tk', '.ga', '.cf', '.ml', '.bit']) else 0
    }
    return features



#data load
data = pd.read_csv('./Datasets/url_data_mega_deep_learning_checked.csv')
data.columns = ['text', 'label']

# plt.figure
# data['label'].value_counts().plot(kind='pie', autopct='%1.0f%%')
# plt.title('Labels')
# plt.show()

vect = TfidfVectorizer()
X = vect.fit_transform(data['text'])
y = data['label']


data = pd.concat([data, data['text'].apply(lambda x: pd.Series(extract_features(x)))], axis=1)
X = data.drop(columns=['text', 'label'])
#print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'MNB': MultinomialNB(),
    'SVM (Linear)': LinearSVC(dual=False)
}

for name, model in models.items():
    print(f"Training mode {name}, standby...")
    start_time = perf_counter()
    model.fit(X_train, y_train)
    end_time = perf_counter()
    time_taken = end_time - start_time
    print(f"Training time: {time_taken:.2f} secs")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Percentage: {accuracy*100:.2f}%")
    print(f"-----------------------------------------")

# model = LinearSVC(dual=False, random_state=42, max_iter=50000)
# final_model = model.fit(X_train, y_train)
# y_pred = final_model.predict(X_test)

# test_url = pd.Series('www.G00gle.com/someevil.php')
# print(final_model.predict(vect.transform(test_url))[0])

