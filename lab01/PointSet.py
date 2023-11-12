from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        label_frequency = [0,0]
        for label in self.labels:
            if label == 0:
                label_frequency[0] += 1
            else:
                label_frequency[1] += 1
        point_count = len(self.labels)
        label_frequency = list(map(lambda x:x/point_count, label_frequency))
        return 1 - label_frequency[0]**2 - label_frequency[1]**2
            

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        n = len(self.labels)
        self.maximizing_feature = {} #This dictionnary enables to save the value of the categorical or real feature that maximizes the gini when used to split
        gini_results = []
        for feature_index in range(len(self.features[0])):
            #We'll use an attribute if the feature is categorical, to store the value such as the associated split maximizes the gini gain
            if self.types[feature_index] == FeaturesTypes.BOOLEAN:    
                child1_features = []
                child2_features = []
                child1_labels = []
                child2_labels = []
                #Split if the feature is boolean
                for i,f in enumerate(self.features):
                    if f[feature_index] == 0:
                        child1_features.append(f)
                        child1_labels.append(self.labels[i])
                    else:
                        child2_features.append(f)
                        child2_labels.append(self.labels[i])
                if len(child1_features) < self.min_split_points or len(child2_features) < self.min_split_points:
                    gini_results.append((feature_index, 0))
                    continue
                child1 = PointSet(child1_features,child1_labels,self.types)
                child2 = PointSet(child2_features,child2_labels,self.types)
                gini1 = child1.get_gini()
                gini2 = child2.get_gini()
                n1 = len(child1.labels)
                n2 = len(child2.labels)
                gini_split = (n1/n)*gini1 + (n2/n)*gini2
                gini_gain = self.get_gini() - gini_split
                gini_results.append((feature_index, round(gini_gain,7)))
            
            elif self.types[feature_index] == FeaturesTypes.CLASSES:
                #Split if the feature is categorical (first find the max value, then do the splits) (we do 1 versus others)
                #We don't save every gain for every value, but only the value of each categorical feature maximizing the gain (so at the end, gini_results contains one value for each feature)
                max_cat = int(max([cat for cat in self.features[:,feature_index]]))
                current_max_gini = 0
                current_value_maximizing = 0
                for cat in range(max_cat+1):
                    child1_features = []
                    child2_features = []
                    child1_labels = []
                    child2_labels = []
                    #We can now do the split
                    for i,f in enumerate(self.features):
                        if f[feature_index] == cat:
                            child1_features.append(f)
                            child1_labels.append(self.labels[i])
                        else:
                            child2_features.append(f)
                            child2_labels.append(self.labels[i])
                    if len(child1_features) < self.min_split_points or len(child2_features) < self.min_split_points:
                        continue
                    child1 = PointSet(child1_features,child1_labels,self.types)
                    child2 = PointSet(child2_features,child2_labels,self.types)
                    gini1 = child1.get_gini()
                    gini2 = child2.get_gini()
                    n1 = len(child1.labels)
                    n2 = len(child2.labels)
                    gini_split = (n1/n)*gini1 + (n2/n)*gini2
                    gini_gain = self.get_gini() - gini_split
                    if gini_gain > current_max_gini:
                        current_value_maximizing = cat
                        current_max_gini = gini_gain
                gini_results.append((feature_index, round(current_max_gini,7)))
                self.maximizing_feature[feature_index] = current_value_maximizing
                
            elif self.types[feature_index] == FeaturesTypes.REAL:
                #Split if the feature is real (according to the method explained in the subject)
                #We don't save every gain for every value, but only the value of each real feature maximizing the gain (exactly as for categorical features)
                continuous_values = sorted(self.features[:,feature_index])
                current_max_gini = 0
                current_value_maximizing = 0
                for value in continuous_values:
                    child1_features = []
                    child2_features = []
                    child1_labels = []
                    child2_labels = []
                    #We can now do the split
                    for i,f in enumerate(self.features):
                        if f[feature_index] < value:
                            child1_features.append(f)
                            child1_labels.append(self.labels[i])
                        else:
                            child2_features.append(f)
                            child2_labels.append(self.labels[i])
                    if len(child1_features) < self.min_split_points or len(child2_features) < self.min_split_points:
                        continue
                    child1 = PointSet(child1_features,child1_labels,self.types)
                    child2 = PointSet(child2_features,child2_labels,self.types)
                    gini1 = child1.get_gini()
                    gini2 = child2.get_gini()
                    n1 = len(child1.labels)
                    n2 = len(child2.labels)
                    gini_split = (n1/n)*gini1 + (n2/n)*gini2
                    gini_gain = self.get_gini() - gini_split
                    if gini_gain > current_max_gini:
                        current_value_maximizing =  (max([child1_features[k][feature_index] for k in range(len(child1_features))]) + min([child2_features[k][feature_index] for k in range(len(child2_features))]))/2
                        current_max_gini = gini_gain
                gini_results.append((feature_index, round(current_max_gini,7)))
                self.maximizing_feature[feature_index] = current_value_maximizing
            
        max_index = 0
        for index in range(len(gini_results)):
            if gini_results[index][1] > gini_results[max_index][1]: max_index = index
        self.deciding_feature = gini_results[max_index][0]
        return gini_results[max_index]


    def get_best_threshold(self) -> float:
        feature_index = self.deciding_feature
        if self.types[feature_index] == FeaturesTypes.BOOLEAN:
            return None
        else:
            return self.maximizing_feature[feature_index] #The categorical and continuous features are handled the same way, with the dictionnary