import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        # raise NotImplementedError
        n = pred.shape[0]
        mse = (1/n) * np.sum(np.square(label - pred))
        return np.sqrt(mse)

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        n = x.shape[0]
        feat = np.ones((n, degree + 1))
        for i in range(degree + 1):
            tempo = np.power(x, i)
            feat[:, i] = tempo
            
        return feat
            

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        return np.dot(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        # raise NotImplementedError
        weight = np.linalg.inv(np.matmul(xtrain.T, xtrain))
        weight = np.matmul(weight, xtrain.T)
        weight = np.matmul(weight, ytrain)
        return weight
        
    # extra credit
    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    # extra credit    
    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        # raise NotImplementedError
        d = xtrain.shape[1]
        a = np.matmul(xtrain.T, xtrain)
        b = np.eye(d) * c_lambda
        weight = np.linalg.inv(np.add(a, b))
        weight = np.matmul(weight, xtrain.T)
        weight = np.matmul(weight, ytrain)
        
        return weight

    # extra credit
    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    # extra credit 
    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        raise NotImplementedError

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args: 
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """    
        # raise NotImplementedError
        
        meanErrors = np.ones(10)
        subs = np.array_split(X, kfold, axis = 0)
        ys = np.array_split(y, kfold, axis = 0)
        for k in range(kfold):
            testing = subs[k]
            
            trainX = np.delete(subs, k, axis = 0)
            training = np.concatenate(trainX)
            
            trainY = np.delete(ys, k, axis = 0)
            trainingY = np.concatenate(trainY)
            
            weight = self.ridge_fit_closed(training, trainingY, c_lambda)
            preds = self.predict(testing, weight)
            meanErrors[k] = self.rmse(preds, ys[k])
            
        meanError = np.average(meanErrors)
        return meanError
            
        

        