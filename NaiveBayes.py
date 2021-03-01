import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        """Calculates basic statistics used in predition:

        Parameters
        ----------
        X : list of list or array 
            explanatory variables in the model
        y : list or array 
            target variables
        """
        self.priors_ = np.bincount(y) / len(y)
        self.means_ = np.array([X[np.where(y==i)].mean(axis=0) for i in np.unique(y)])
        self.stds_ = np.array([X[np.where(y==i)].std(axis=0) for i in np.unique(y)])
        self.y_classes = np.unique(y) 
        
    def predict_proba(self, X):
        """Calcs posterior probability for each class in y, multiplies by prior probability (of each class)
        
        Parameters
        ----------
        X : list or array of explanatory variables (list)

        Returns : array of probabilities
        """
        results = []
        for x in X:
            probabilities = []
            for j in self.y_classes:
                y_i = np.where(self.y_classes==j)
                probabilities.append(self.gauss(x, self.stds_[y_i], self.means_[y_i]).prod()*self.priors_[y_i][0])

            probabilities = np.array(probabilities)
            results.append(probabilities / probabilities.sum())
        return np.array(results)

    def gauss(self, x, std, mean):
        """Computes gaussian probability given an x, variance, and a mean
        
        Parameters:
        ----------
        x : list/array of explanatory variables
        std : standard deviation
        mean : mean 

        Returns:
        --------
        float
        """
        return 1 / np.sqrt( 2 * np.pi * std**2 ) * np.exp(-0.5*( ( x- mean ) / std)**2 )

    def predict(self, X):
        """Predict classes of array X.

        Parameters
        ----------
        X : array/list of lists 

        Returns
        -------
        classes : predicted classes
        """
        probabilities = self.predict_proba(X)
        return probabilities.argmax(axis=1)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=20, centers=[(2,3), (-2,-5), (-4, 2)], random_state=1)


    # test with sklearn model:
    my_model = GaussianNaiveBayes()
    my_model.fit(X, y)
    print("My model:")
    print(my_model.predict_proba([[1, 1], [-2, 3], [-1.5, -3]]))
    print(my_model.predict([[1, 1], [-2, 3], [-1.5, -3]]))

    # sklearn model:
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X, y)
    print("")
    print("SKLearn:")
    print(gnb.predict_proba([[1, 1], [-2, 3], [-1.5, -3]]))
    print(gnb.predict([[1, 1], [-2, 3], [-1.5, -3]]))