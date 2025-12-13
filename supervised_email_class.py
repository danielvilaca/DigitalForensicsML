import pandas as pd
from time import perf_counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

#importing email data
df = pd.read_csv('./enron_spam_data_label.csv')
print(df.shape)
df = df.dropna()
print(df.shape)

#extracting features
cvect = CountVectorizer()
X = cvect.fit_transform(df.Message)
y = df.label_num

#email data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "MultinomialNB": MultinomialNB(),
    "SVM (Linear)": SVC()
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
