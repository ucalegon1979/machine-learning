from ojapca import ojaPCA
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing


def plot_objectives(objectives):
    fig, ax = plt.subplots()
    
    for obj in objectives:
        ax.plot(obj[0:10])
        
    plt.xlabel('Iteration')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of PCA Fit')
    
    plt.show()
    

train = pd.read_csv("mnist_train.csv", header=None)

# the first column are the labels of each picture
y = np.asarray(train.ix[:, 0])
# the rest of the columns are the features
X = train.ix[:, 1:]

#standardize the features
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

pca = ojaPCA()
pca.fit(X)
print('My first principal component:', pca.components_[0])
print('My second principal component:', pca.components_[1])
print('My third principal component:', pca.components_[2])



plot_objectives(pca.eigvals_)

def plot_objectives(objectives):
    fig, ax = plt.subplots()
    
    for obj in objectives:
        ax.plot(obj[0:10])
        
    plt.xlabel('Iteration')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of PCA Fit')
    
    plt.show()
    

train = pd.read_csv("mnist_train.csv", header=None)

# the first column are the labels of each picture
y = np.asarray(train.ix[:, 0])
# the rest of the columns are the features
X = train.ix[:, 1:]

#standardize the features
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

pca = ojaPCA()
pca.fit(X)
print('My first principal component:', pca.components_[0])
print('My second principal component:', pca.components_[1])
print('My third principal component:', pca.components_[2])



plot_objectives(pca.eigvals_)





# Now compare with scikit-learn!
from sklearn.decomposition import PCA

# Compare to sklearn
pca = PCA(5, svd_solver='full')
pca.fit(X)
print('First principal component:', pca.components_[0])
print('Second principal component:', pca.components_[1])
print('Third principal component:', pca.components_[2])