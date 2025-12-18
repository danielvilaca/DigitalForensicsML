import numpy as np
import pandas as pd
import math
from collections import Counter

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer #better than CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from time import perf_counter
from sklearn.metrics import accuracy_score, confusion_matrix

from urllib.parse import urlparse, parse_qs
import re


def calculate_entropy(text):
    if not text:
        return 0
    probs = [n_c / len(text) for n_c in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs)

def extract_features(url):
    parsed_url = urlparse(url)
    host = parsed_url.netloc if parsed_url.netloc else url.split('/')[0]
    features = {
        'url_length': len(url),
        'hostname_length': len(host),
        'path_length': len(parsed_url.path),
        'entropy': calculate_entropy(url),
        'digits_count': sum(c.isdigit() for c in url),
        'special_chars': sum(c in ['-', '_', '@', '/', '?', '=', '.', '%', '&', '!', ';', '+'] for c in url),
        'subdomain_count': host.count('.'),
        'has_https': 1 if url.startswith("https") else 0,
        'has_ip': 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0,
        'has_at_symbol': 1 if '@' in url else 0,
        'suspicious_words': 1 if any(word in url.lower() for word in ['login', 'verify', 'bank', 'secure', 'update']) else 0,
        'suspicious_tlds': 1 if any(url.endswith(tld) for tld in ['.xyz', '.top', '.pw', '.tk', '.ga', '.cf', '.ml', '.bit', '.zip']) else 0
    }
    return features



#data load
data = pd.read_csv('./Datasets/url_data_mega_deep_learning_checked.csv')
data.columns = ['text', 'label']
y = data['label']

# plt.figure
# data['label'].value_counts().plot(kind='pie', autopct='%1.0f%%')
# plt.title('Labels')
# plt.show()

print("Vectorizing text (TF-IDF)...")
vect = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
X_tfidf = vect.fit_transform(data['text'])


#data = pd.concat([data, data['text'].apply(lambda x: pd.Series(extract_features(x)))], axis=1)
data = data['text'].apply(lambda x: pd.Series(extract_features(x)))

scaler = StandardScaler()
X_manual_scaled = scaler.fit_transform(data)

X_final = hstack([X_tfidf, X_manual_scaled])

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, shuffle=True, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=30),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM (Linear)': LinearSVC(dual=False)
}

# for name, model in models.items():
#     print(f"Training mode {name}, standby...")
#     start_time = perf_counter()
#     model.fit(X_train, y_train)
#     end_time = perf_counter()
#     time_taken = end_time - start_time
#     print(f"Training time: {time_taken:.2f} secs")
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy Percentage: {accuracy*100:.2f}%")
#     print(f"-----------------------------------------")


model = RandomForestClassifier()
final_model = model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


test_url = 'www.G00gle.com/someevil.php'
single_df = pd.DataFrame({'text': [test_url]})
single_tfidf = vect.transform(single_df['text'])
single_manual = single_df['text'].apply(lambda x: pd.Series(extract_features(x)))
single_manual_scaled = scaler.transform(single_manual)
single_final_features = hstack([single_tfidf, single_manual_scaled])

prediction = final_model.predict(single_final_features)[0]
print(f"\nURL: {test_url}")
print(f"Prediction: {prediction}")


#Confusion Matrix - Matplot + Seaborn
# cols = list(final_model.classes_)
# ax = plt.subplot()
# CM_LR = confusion_matrix(y_test, y_pred)
# sns.heatmap(CM_LR, annot=True, fmt='.1f', ax=ax, cmap='RdBu')
# ax.set_xlabel('Predicted Labels')
# ax.set_ylabel('True Labels')
# ax.set_title(f'Confusion Matrix - {model}')
# ax.xaxis.set_ticklabels(cols)
# ax.yaxis.set_ticklabels(cols)
# plt.show()
