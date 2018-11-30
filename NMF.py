class my_NMF():
    
    def __init__(self, n_components, loss = 'Frobenius', epsilon = 1.5*10e-3, max_iter = 100):
        '''
        Attributes:
        
        n_components_ : integer
            the unknown dimension of W and H
        loss_ = {"Frobenius", "KL"}
        max_iter_: integer
            maximum number of iterations
        epsilon_: float
            convergence
        w_: np.array
            W Matrix factor
        H_: np.array
            H Matrix factor
        '''
        self.n_components_ = n_components
        self.max_iter_ = max_iter
        self.loss_ = loss
        self.epsilon_ = epsilon
        self.W_ = None
        self.H_ = None
        
    def KL_divergence(self,M1,M2):
        print('Computing the KL divergence ..')
        return np.sum(np.sum([M1*np.log(M1/(M2+10e-8))-M1+M2],axis = 1),axis = -1)[0]
        
    def fit_transform(self, X, sparse):
        """ Find the factor matrices W and H
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self
        """    
        n = X.shape[0]
        p = X.shape[1]
        
        X = np.transpose(X)

        #We initialize randomly W (p,r) and H (r,n)
        if sparse == True :
            W,H = scipy.sparse.random(X.shape[0],self.n_components_),scipy.sparse.random(self.n_components_,X.shape[1])
            if self.loss_ == "Frobenius":

                W_old,H_old = W * 100 , H * 100
                
                iter_ = 0
                
                while scipy.sparse.linalg.norm(W@H - W_old@H_old,ord = 'fro') > self.epsilon_ or iter_ < self.max_iter_:
                  
                    W_old = W
                    H_old = H

                    H *= np.divide((W.T)@X,(W.T)@(W@H),where = (W.T)@(W@H) != 0)
                    # W will use the new value of H
                    W *= np.divide(X@(H.T),(W@H)@(H.T),where = ((W@H)@(H.T)) != 0)
            
            if self.loss_ == "KL":
              
                
                W_old,H_old = W * 100 , H * 100
                
                iter_ = 0
                
                while iter_ == 0 or self.KL_divergence(W_old@H_old,WH) > self.epsilon_ or iter_ < self.max_iter_:
                    W_old = W
                    H_old = H
                    
                    WH = W@H
                    X_WH = np.divide(X,WH,where = WH != 0)
                    
                    H *= (np.divide((X_WH.T)@W , W.sum(axis=0),where = W.sum(axis = 0) != 0).T)
                    
                    
                    WH = W@H
                    X_WH = np.divide(X,WH,where = WH != 0)
                    W *= np.divide(X_WH@(H.T), H.sum(axis=1),where = H.sum(axis=1) != 0)
                    
                    WH = W.dot(H)
                    
                    print('The KL divergence is ',self.KL_divergence(W_old@H_old,WH))
                    
                    iter_ += 1
              
        else:
            W,H = np.random.rand(X.shape[0],self.n_components_),np.random.rand(self.n_components_,X.shape[1])
            
            if self.loss_ == "Frobenius":
                W_old,H_old = W * 100 , H * 100
                
                iter_ = 0
                
                while scipy.linalg.norm(W@H - W_old@H_old ,ord = 'fro') > self.epsilon_ or iter_ < self.max_iter_:
                    W_old = W
                    H_old = H

                    H *= np.divide((W.T)@X,(W.T)@(W@H),where = (W.T)@(W@H) != 0)
                    # W will use the new value of H
                    W *= np.divide(X@(H.T),(W@H)@(H.T),where = ((W@H)@(H.T)) != 0)
                    
                    iter_ += 1
            
            elif self.loss_ == "KL":
              
                W_old,H_old = W * 100 , H * 100
                iter_ = 0
                
                while iter_ == 0 or self.KL_divergence(W_old@H_old,WH) > self.epsilon_ or iter_ < self.max_iter_:
                    W_old = W
                    H_old = H
                    
                    WH = W@H
                    X_WH = np.divide(X,WH,where = WH != 0)
                    
                    H *= (np.divide((X_WH.T)@W , W.sum(axis=0),where = W.sum(axis = 0) != 0).T)
                    
                    WH = W@H
                    X_WH = np.divide(X,WH,where = WH != 0)
                    W *= np.divide(X_WH@(H.T), H.sum(axis=1),where = H.sum(axis=1) != 0)
                    
                    WH = W.dot(H)
                    
                    print('The KL divergence is ',self.KL_divergence(W_old@H_old,WH))
                    
                    iter_ += 1
                    
        self.W_ = W
        self.H_ = H