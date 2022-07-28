from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
import numpy as np

class MetaExplain:
    def __init__ (self, train_feature_importance, feature_names):
        self.feature_names = feature_names
        #if train_feature_importance != None:
        #    self.train_feature_importance = train_feature_importance
        #    self.shape = train_feature_importance.shape
        #    self.nn = NearestNeighbors()
        #    self.nn.fit(train_feature_importance.reshape((len(train_feature_importance),self.shape[1]*self.shape[2])))

        #self.global_PCA = self._initialize_unsupervised(train_feature_importance, 'PCA')
        #self.global_KPCA = self._initialize_unsupervised(train_feature_importance, 'KPCA')
        #self.global_DR = {'PCA': self.global_PCA, 'KPCA': self.global_KPCA}

    def _initialize_unsupervised(self, feature_importance, type): #Global initialize once with train for type PCA, KPCA
        DR = []
        for i in range(len(self.feature_names)):
            temp = feature_importance[:,:,i]
            if type == "KPCA":
                DR.append(KernelPCA(n_components=1, kernel='rbf', random_state=i).fit(temp))
            else:
                DR.append(PCA(n_components=1, random_state=i).fit(temp))
        return DR
    
    def _find_neighbours(self, feature_importance, k):
        idx = self.nn.kneighbors(feature_importance.reshape((1,self.shape[1]*self.shape[2])), k, return_distance=False)
        neighbours = []
        for i in idx[0]:
            neighbours.append(self.train_feature_importance[i])
        neighbours.append(feature_importance.reshape((1,self.shape[1],self.shape[2]))[0])
        return np.array(neighbours).reshape((k+1,self.shape[1],self.shape[2]))
    
    def meta_unsupervised(self, feature_importance, type='KPCA', scope='global', k = 50):
        if scope == 'global':
            DR = self.global_DR[type].copy()
            new_feature_importance = []
            for i in range(len(self.feature_names)):
                temp = feature_importance[:,1]
                new_feature_importance.append(DR[i].transform(np.array([temp,temp]))[0])
            return np.array(new_feature_importance)
        else:
            neighbours = self._find_neighbours(feature_importance, k)
            local_DR = self._initialize_unsupervised(neighbours, type)
            new_feature_importance = []
            for i in range(len(self.feature_names)):
                temp = feature_importance[:,i]
                new_feature_importance.append(local_DR[i].transform(np.array([temp,temp]))[0])
            return np.array(new_feature_importance)

    def meta_avg(self, feature_importance):
        return feature_importance.mean(axis=0)

    def meta_median(self, feature_importance):
        return np.median(feature_importance, axis=0)

    def meta_rule_based(self, truthfulness, average_change, feature_importance):
        _, l2 = zip(*sorted(zip(average_change, range(len(average_change))), reverse=True))

        candidate_fis = {}
        count = 0
        for feature in self.feature_names:
            temp = []
            for f in range(len(feature_importance)):
                if feature not in truthfulness[f]:
                    temp.append(feature_importance[f][count])
            candidate_fis[feature] = temp
            count = count + 1
        newmax = None
        meta_interpretation = {}
        
        for j in l2:
            feature = self.feature_names[j]
            if len(candidate_fis[feature])!=0:
                if newmax is None: #from the most important feature take the higher abs importance value 
                    indmax = np.argmax([abs(i) for i in candidate_fis[feature]])
                    newmax = abs(candidate_fis[feature][indmax])
                else:
                    if len(candidate_fis[feature]) == 1: #if only one 
                        indmax = 0
                        newmax = min(newmax,abs(candidate_fis[feature][indmax]))
                    else: #if more than one, then take the higher importance value, but if it is possible to be lower than the previously added importance
                        t1, t2 = zip(*sorted(zip([abs(kk) for kk in candidate_fis[feature]], range(len(candidate_fis[feature]))), reverse=True))
                        finish = True
                        found = False
                        ct = 0
                        while finish:
                            if t1[ct] < newmax:
                                indmax = t2[ct]
                                newmax = t1[ct]
                                finish = False
                                found = True
                            ct = ct + 1
                            if ct == len(t1):
                                finish = False
                        if not found:
                            indmax = t2[-1]
                            newmax = min(newmax,t1[-1])
                meta_interpretation[feature] = candidate_fis[feature][indmax]
            else:
                meta_interpretation[feature] = 0

        final_meta_interpretation = []
        for feature in self.feature_names:
            if feature in meta_interpretation:
                final_meta_interpretation.append(meta_interpretation[feature])
            else:
                final_meta_interpretation.append(0)
        return np.array(final_meta_interpretation)