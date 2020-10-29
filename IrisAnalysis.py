import numpy as np
from sklearn import datasets

# Iris dataset is already in scikit-learn :)
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class lebels:', np.unique(y))

"""Should print te following: 'Class labels: [0 1 2]'

The labels 0, 1, and 2 represent the three typs of iris flower, namely
setosa, versicolor, and virginica."""

# The train_test_split function randomly splits X, y into a specified amout of training and test data
from sklearn.model_selection import train_test_split

# Here, we dictate that 30% of the data will be for testing, and 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Time to standardize the data (feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# We're going to import an train a perceptron for multiclass classification (with One-versus-Rest method)
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)