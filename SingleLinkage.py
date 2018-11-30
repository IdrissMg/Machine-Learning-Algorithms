class my_SingleLinkageAglomerativeClustering():
    
    def __init__(self, metric="euclidean", n_clusters=3):
        '''
        Attributes:
        
        metric_: {"euclidean","precomputed"}
            the distance to be used
            if precomputed then X is distance matrix
        n_clusters: integer
            number of clusters to return 
        linkage_matrix_: (n-1, 4) np.array
            in the same format as linkage  
        labels_: integer np.array
            label assigment
        hierarchy_: list of np.array
            each array corresponds to label assigment
            at each level (number of clusters)
            hierarchy_[0]=np.array(list(range(n)))
        '''
        self.metric_ = metric
        self.n_clusters_ = n_clusters
        self.linkage_matrix_ = None
        self.labels_ = None
        self.hierarchy_ = None
        
        
    def smallest_distance(self, dist_matrix):
        """
        returns the smallest distance in the distance matrix
        """
        
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape) 
        
        return dist_matrix[i, j], i, j
    
    def single_linkage_update(self,i,j,dist_matrix):
        """
        updates the distance matrix using the single linkage technique
        """
        clusters_list = np.array(range(dist_matrix.shape[0]))
        clusters_filtered = clusters_list[(clusters_list != i) & (clusters_list != j)]
        
        #The proximity between the new cluster, denoted (r,s) and old cluster (k) is defined as :
        #            d[(k), (r,s)] = min d[(k),(r)], d[(k),(s)]
            

        update = np.minimum(dist_matrix[i,:],dist_matrix[j,:])
        n = dist_matrix.shape[0]
        
        tmp = np.zeros((n+1,n+1))
        tmp[:-1,:-1] = dist_matrix
        tmp[-1,:],tmp[:,-1] = np.append(update,10e5),np.append(update,10e5)
        
        tmp[i,:] = [10e5]*(n+1)
        tmp[:,i] = [10e5]*(n+1)
        
        tmp[j,:] = [10e5]*(n+1)
        tmp[:,j] = [10e5]*(n+1)
        
        
        dist_matrix = tmp

        return dist_matrix
        
    def fit(self, X):
        """ Create a hierarchy of clusters
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        self: my_SingleLinkageAglomerativeClustering
            to have access to labels_
        """
        
        if self.metric_ == 'euclidean':
            X_ = distance_matrix(X,X)
            for i in range(len(X_)):
                X_[i,i] = 10e5
        else:
            X_ = X.copy()
            
        dist_matrix = X_.copy()
        
        self.linkage_matrix_ = []
        
        n = dist_matrix.shape[0]
        
        #history will contain the elements in each cluster / labels will contain the cluster of each element
        self.history,self.labels = {},{}
        for i in range(n):
            self.history[i] = [i]
            self.labels[i]  = i
               
        while len(dist_matrix) < 2*n - 1: #We stop when we have only one cluster : n (initially)+ n-1 (maximum of clusters built during the process)
                     
            min_distance, i, j = self.smallest_distance(dist_matrix)
            
            dist_matrix = self.single_linkage_update(i,j,dist_matrix)
            
            self.labels[i],self.labels[j] = dist_matrix.shape[0]-1,dist_matrix.shape[0]-1
            
            self.history[dist_matrix.shape[0]-1] = self.history[i] + self.history[j]
            
            for ix in [i,j]:
              if len(dist_matrix) <= 2*n -1 - self.n_clusters_ :
                for elem in self.history[ix]:
                  self.labels[elem] = self.labels[ix]
            
            self.linkage_matrix_.append([i,j,min_distance,len(self.history[dist_matrix.shape[0]-1])])
    
        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        self.labels = [self.labels[i] for i in range(n)] 
            
            
    def plot_dendrogram(self,truncate_mode_=None,p_=30):
        '''Use self.linkage_matrix_ in `scipy.cluster.hierarchy.dendrogram` 
        to plot the dendrogram of the hierarchical structure
        '''      
        hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
        fig, axes = plt.subplots(1, 1, figsize=(20, 15))
        self.dn = hierarchy.dendrogram(self.linkage_matrix_,ax=axes,p = p_, truncate_mode = truncate_mode_,orientation='top')
        hierarchy.set_link_color_palette(None)  # reset to default after use
        plt.show()
        