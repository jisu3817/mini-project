```python
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
url = 'https://github.com/dknife/ML/raw/main/data/Proj2/faces/'

face_images = []

for i in range(15):
    file = url + 'img{0:02d}.jpg'.format(i+1)
    img = imread(file)
    img = resize(img, (64,64))
    face_images.append(img)

def plot_images(nRow, nCol, img):
    fig = plt.figure()
    fig, ax = plt.subplots(nRow, nCol, figsize = (nCol,nRow))
    for i in range(nRow):
        for j in range(nCol):
            if nRow <= 1: axis = ax[j]
            else:         axis = ax[i, j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(img[i*nCol+j])

plot_images(3,5, face_images)
```

![스크린샷 2022-11-10 오전 12.20.30.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4934dc33-b116-4d49-8968-289be7d70673/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.20.30.png)

```python
face_hogs = []
face_features = []

for i in range(15):
    hog_desc, hog_image = hog(face_images[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    face_hogs.append(hog_image)
    face_features.append(hog_desc)

plot_images(3, 5, face_hogs)

print(face_features[0].shape)

fig = plt.figure()
fig, ax = plt.subplots(3,5, figsize = (10,6))
for i in range(3):
    for j in range(5):
        ax[i, j].imshow(resize(face_features[i*5+j], (128,16)))
```

결과 : (128,)

![스크린샷 2022-11-10 오전 12.21.23.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/35e6bde8-2ed3-4a40-8ffb-c6cfc69a5830/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.21.23.png)

```python
url = 'https://github.com/dknife/ML/raw/main/data/Proj2/animals/'

animal_images = []

for i in range(15):
    file = url + 'img{0:02d}.jpg'.format(i+1)
    img = imread(file)
    img = resize(img, (64,64))
    animal_images.append(img)

plot_images(3, 5, animal_images)
```

![스크린샷 2022-11-10 오전 12.22.02.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5473d49b-c30c-4a3e-97a2-550030952137/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.22.02.png)

```python
animal_hogs = []
animal_features = []

for i in range(15):
    hog_desc, hog_image = hog(animal_images[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    animal_hogs.append(hog_image)
    animal_features.append(hog_desc)

plot_images(3, 5, animal_hogs)

fig = plt.figure()
fig, ax = plt.subplots(3,5, figsize = (10,6))
for i in range(3):
 for j in range(5):
   ax[i, j].imshow(resize(animal_features[i*5+j], (128,16)))
```

![스크린샷 2022-11-10 오전 12.22.28.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/273283f2-fbe8-478f-90b2-0422180c845a/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.22.28.png)

```python
X, y = [], []

for feature in face_features:
    X.append(feature)
    y.append(1)
for feature in animal_features:
    X.append(feature)
    y.append(0)

fig = plt.figure()
fig, ax = plt.subplots(6,5, figsize = (10,6))
for i in range(6):
 for j in range(5):
   ax[i, j].imshow(resize(X[i*5+j], (128,16)),interpolation='nearest')
print(y)
```

결과 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

![스크린샷 2022-11-10 오전 12.23.46.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3aaac547-752b-421c-9fe5-581f7f6eee66/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.23.46.png)

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

polynomial_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(C=1, kernel = 'poly', degree=5, coef0=10.0))
 ])
polynomial_svm_clf.fit(X, y)
```

결과

```python
Pipeline(memory=None,
         steps=[('scaler',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('svm_clf',
                 SVC(C=1, break_ties=False, cache_size=200, class_weight=None,
                     coef0=10.0, decision_function_shape='ovr', degree=5,
                     gamma='scale', kernel='poly', max_iter=-1,
                     probability=False, random_state=None, shrinking=True,
                     tol=0.001, verbose=False))],
         verbose=False)
```

```python
yhat = polynomial_svm_clf.predict(X)
print(yhat)
```

결과: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

```python
url = 'https://github.com/dknife/ML/raw/main/data/Proj2/test_data/'

test_images = []

for i in range(10):
    file = url + 'img{0:02d}.jpg'.format(i+1)
    img = imread(file)
    img = resize(img, (64,64))
    test_images.append(img)

plot_images(2, 5, test_images)
```

![스크린샷 2022-11-10 오전 12.25.31.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6645f4a4-d004-4083-ae7e-9797e7adcd54/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.25.31.png)

```python
test_hogs = []
test_features = []
for i in range(10):
    hog_desc, hog_image = hog(test_images[i], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    test_hogs.append(hog_image)
    test_features.append(hog_desc)

plot_images(2, 5, test_hogs)

fig = plt.figure()
fig, ax = plt.subplots(2,5, figsize = (10,4))
for i in range(2):
 for j in range(5):
   ax[i, j].imshow(resize(test_features[i*5+j], (128,16)), interpolation='nearest')
```

![스크린샷 2022-11-10 오전 12.25.54.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eded026e-8c24-4deb-a26b-367307fbe41d/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.25.54.png)

```python
test_result = polynomial_svm_clf.predict(test_features)
print(test_result)
```

결과: [1 0 1 0 0 0 0 0 1 0]

```python
fig = plt.figure()
fig, ax = plt.subplots(2,5, figsize = (10,4))
for i in range(2):
    for j in range(5):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
        if test_result[i*5+j] == 1:
            ax[i, j].imshow(test_images[i*5+j],interpolation='nearest')
```

![스크린샷 2022-11-10 오전 12.26.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/829bb841-7cac-402c-baec-6b329451ca40/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-10_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.26.44.png)
