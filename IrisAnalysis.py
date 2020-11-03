import numpy as np
from sklearn import datasets

# Iris dataset is already in scikit-learn :)
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('Class labels:', np.unique(y))

"""Should print the following: 'Class labels: [0 1 2]'

The labels 0, 1, and 2 represent the three typs of iris flower,
setosa, versicolor, and virginica."""

# The train_test_split function randomly splits X, y into a specified amout of training and test data
from sklearn.model_selection import train_test_split

# Here, we put aside 30% of the data for testing and 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Time to standardize the data (feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# The fit method estimates the sample mean and standard deviation for each feature dimension
sc.fit(X_train)
# The transform method stardardizes the data using the estimated sample mean and standard deviation
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# We're going to import an train a perceptron for multiclass classification (with One-versus-Rest method)
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=42, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# Now we can make predictions
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' %(y_test != y_pred).sum())

"""Notes about the Perceptron.fit method and the results: 
 - I got a deprecation error for n_iter (which the book uses), so I changed it to max_iter.
 - For both n_iter = 40 and max_iter = 40, I get Misclassified samples = 9.
 - For max_iter = 41, I get Misclassified samples = 6.
 - For max_iter = 42, I get Misclassified samples = 2.
 - I did not use the tol parameter in the fit function because it defaults to None (aka -infinity).
 - Even if I set tol = 1e-12, the Misclassified samples = 14, so I let tol = None.
"""

# Calculates the classification accuracy of the perceptron on the test data:
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' %accuracy_score(y_test, y_pred))

# Alternative way to check accuracy:
print('Accuracy: %.2f' %ppn.score(X_test_std, y_test))

# Plotting with DecisionRegionPlotter
from DecisionRegionPlotter import *
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

"""Note about final DecisionRegion Plot:
The three flower classes cannot be perfectly separated by a linear decision boundary."""