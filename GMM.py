class my_GMM():
    
    def __init__(self, k, initialization="kmeans"):
        '''
        Attributes:
        
        k_: integer
            number of components
        initialization_: {"kmeans", "random"}
            type of initialization
        mu_: np.array
            array containing means
        Sigma_: np.array
            array cointaining covariance matrix
        cond_prob_: (n, K) np.array
            conditional probabilities for all data points 
        labels_: (n, ) np.array
            labels for data points
        '''
        self.k_ = k
        self.initialization_ = initialization
        self.mu_ = None
        self.Sigma_ = None
        self.cond_prob_ = None
        self.labels_ = None
        
    def compute_condition_prob_matrix(self,X, p ,mu, Sigma):
        '''Compute the conditional probability matrix 
        shape: (n, K)
        '''
        n = X.shape[0]
        prob_matrix = np.zeros((n,self.k_))
        for i in range(len(prob_matrix)):
            prob_matrix[i,:] = [p[k]*self.gauss_density(X[i],mu[k],Sigma[k])
                                /np.sum([p[k]*self.gauss_density(X[i],mu[k],Sigma[k]) for k in range(self.k_)]) 
                                for k in range(self.k_)]
        return prob_matrix
    
    def gauss_density(self,x,mu,sigma):
        '''
        Multivariate Gaussian density function
        x : vector
        mu : vector
        sigma : covariance matrix
        '''
        a = exp(-1/2 * ((x-mu).reshape((1,-1))).dot(np.linalg.inv(sigma)).dot((x-mu).reshape((-1,1))))
        b = 1/((2*np.pi)**(len(x)/2)*scipy.linalg.det(sigma)**(1/2))
        return  b * a

    def fit(self, X):
        """ Find the parameters mu_ and Sigma_
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self
        """

        def compute_expectation(data_x):
            '''Compute the expectation to check increment'''
            
            n = data_x.shape[0]
            
            if self.initialization_ == 'kmeans':
                
                kM = KMeans(n_clusters = self.k_)
                kM.fit(data_x)
                
                self.labels_ = kM.labels_
                
                p = [Counter(self.labels_)[i]/len(data_x) for i in np.unique(self.labels_)]
                
                mu = kM.cluster_centers_
                
                Sigma = np.array([1/(n*p[k])*
                         np.sum([((data_x[ix]-mu[k]).reshape((-1,1))).dot((data_x[ix]-mu[k]).reshape((1,-1))) 
                         for ix in np.argwhere(np.array(self.labels_)==k).ravel()],axis = 0) 
                         for k in range(self.k_)])
                
            else: #Random initialization
                p = [1/self.k_]*self.k_
                mu = np.random.rand(self.k_,data_x.shape[1])
                Sigma = np.random.rand(self.k_,data_x.shape[1],data_x.shape[1])

            iter_ = 0
    
            while iter_ == 0 or np.sum([np.linalg.norm(old_mu[k] - mu[k]) for k in range(self.k_)]) > 10e-2 :
            
                old_mu = mu
                
                cond_prob = self.compute_condition_prob_matrix(data_x, p ,mu, Sigma)
                
                p = [1/n*np.sum(cond_prob[:,k]) for k in range(self.k_)]
                
                mu = [1/np.sum([cond_prob[i,k] for i in range(n)],axis=0)*\
                      np.sum([data_x[i]*cond_prob[i,k] for i in range(n)],axis = 0) for k in range(self.k_)]
                
                Sigma = [1/np.sum([cond_prob[i,k] for i in range(n)],axis=0)*\
                         np.sum([((data_x[i]-mu[k]).reshape((-1,1))).dot((data_x[i]-mu[k]).reshape((1,-1)))*cond_prob[i,k] 
                                 
                     for i in range(n)],axis = 0) 
                     for k in range(self.k_)]
                
                iter_ += 1
                
            return cond_prob,mu,Sigma,p
        
        self.cond_prob_,self.mu_,self.Sigma_,self.proportion_ = compute_expectation(X)
        
        self.labels_ = [np.argmax(self.cond_prob_[i,:]) for i in range(X.shape[0])]

        
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        prob_matrix_test = self.compute_condition_prob_matrix(X,self.proportion_,self.mu_,self.Sigma_)
        
        labels_test = np.argmax(prob_matrix_test,axis = 1)
        
        return labels_test
        
    def predict_proba(self, X):
        """ Predict probability vector for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        proba: (n, k) np.array        
        """
        prob_matrix_test = self.compute_condition_prob_matrix(X,self.proportion_,self.mu_,self.Sigma_)
        
        return prob_matrix_test