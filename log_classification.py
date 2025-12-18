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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'MNB': MultinomialNB(),
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

