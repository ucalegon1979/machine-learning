from oja-pca import ojaPCA

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