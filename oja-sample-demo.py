from oja-pca import ojaPCA

def plot_objectives(objectives):
    fig, ax = plt.subplots()
    
    for obj in objectives:
        ax.plot(obj[0:10])
        
    plt.xlabel('Iteration')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of PCA Fit')
    
    plt.show()
    



X = np.zeros((250, 50))
X[0:50, :] = np.random.normal(scale=1, size=(50, 50))
X[50:100, :] = np.random.normal(loc=10, scale=5, size=(50, 50))
X[100:150, :] = np.random.normal(loc=65, scale=2, size=(50, 50))
X[150:200, :] = np.random.normal(loc=30, scale=7, size=(50, 50))
X[200:250, :] = np.random.normal(loc=14, scale=3, size=(50, 50))
classes = [0]*50 + [1]*50 + [2]*50 + [3]*50 + [4]*50

pca = ojaPCA()
pca.fit(X)
print('My first principal component:', pca.components_[0])
print('My second principal component:', pca.components_[1])
print('My third principal component:', pca.components_[2])



plot_objectives(pca.eigvals_)