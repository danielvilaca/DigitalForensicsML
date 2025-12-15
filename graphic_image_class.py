import numpy as np

import os
import matplotlib.pyplot as plt

from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

#image processing steps
#resize -> transform grayscale -> HOG feature extraction -> feature scaling

#resizing images
def Resize_images(src, include, width, height=None):

    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    for subdir in os.listdir(src):
        if subdir in include:
            print(f"Processing images in folder: {subdir} standby...")
            current_path = os.path.join(src, subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg'}:
                    im = Image.open(os.path.join(current_path, file))
                    im = im.convert('RGB')
                    if height is not None:
                        im = im.resize((width, height))
                    else:
                        im = im.resize((width, width))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(np.array(im))
    return data

#transform to grayscale
class RGB2GrayTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([rgb2gray(img) for img in X])

#HOG feature extraction
class HOGTransform(BaseEstimator, TransformerMixin):
    def __init__(self, y=None, orientations=8, pixels_per_cell=(10,10), cells_per_block=(1,1), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def local_hog(X):
            return hog(X, orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
        try: #paralell
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

#main
data_path = '/Users/vilaka/Desktop/Datasets/Images/Image'
width = 80
classes = {'other', 'knives', 'guns'}

data = Resize_images(src=data_path, width=width, include=classes)

X = np.array(data['data'])
y = np.array(data['label'])

print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

#image processing pipeline
Image_Pipeline = Pipeline([
    ('Gray_Transform', RGB2GrayTransform()),
    ('HOG_Transform', HOGTransform()),
    ('Scaler', StandardScaler()),
    ('Model', LinearSVC(dual=False, random_state=42, max_iter=50000, C=0.1))
])

print("Running full pipeline")
Image_Pipeline.fit(X_train, y_train)

#teste model accuracy
print("Testing Accuracy")
y_pred = Image_Pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Percentage: {accuracy*100:.2f}%")
