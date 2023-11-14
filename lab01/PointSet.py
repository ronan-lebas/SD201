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
        self.min_split_points = 1

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
            if self.types[feature_index] == FeaturesTypes.BOOLEAN: #Running time is O(n) (no optimization needed) 
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
                #We use an optimization similar to the one for real features : we count the labels for each possible value,
                #and then we compute the gini with the counts stored in dictionnaries, which allows a running time between O(n) and O(n^2) (depends on the sum() Python function) (so Q6 is done much faster)
                max_cat = int(max([cat for cat in self.features[:,feature_index]]))
                current_max_gini = 0
                current_value_maximizing = 0
                gini_start = self.get_gini()
                label_counter_true = {k: 0 for k in range(max_cat+1)}
                label_counter_false = {k: 0 for k in range(max_cat+1)}
                for idx, point in enumerate(self.features):
                    if self.labels[idx]:
                        label_counter_true[point[feature_index]] += 1
                    else:
                        label_counter_false[point[feature_index]] += 1
                #Now we determine the best split
                for value in range(max_cat+1):
                    n1 = label_counter_true[value] + label_counter_false[value]
                    n2 = n - n1
                    if n1 < self.min_split_points or n2 < self.min_split_points:
                        continue
                    gini1 = 1 - ((label_counter_true[value])**2 + (label_counter_false[value])**2)/(n1**2)
                    gini2 = 1 - ((sum(label_counter_true.values())-label_counter_true[value])**2 + (sum(label_counter_false.values())-label_counter_false[value])**2)/(n2**2)
                    gini_split = (n1/n)*gini1 + (n2/n)*gini2
                    gini_gain = gini_start - gini_split
                    if gini_gain > current_max_gini:
                        current_value_maximizing = value
                        current_max_gini = gini_gain
                gini_results.append((feature_index, round(current_max_gini,7)))
                self.maximizing_feature[feature_index] = current_value_maximizing
                
            elif self.types[feature_index] == FeaturesTypes.REAL:
                #Split if the feature is real (according to the method explained in the subject)
                #We use a method get_gini_2
                #We don't save every gain for every value, but only the value of each real feature maximizing the gain (exactly as for categorical features)
                continuous_values = sorted([[self.features[k][feature_index], self.labels[k]] for k in range(len(self.features))], key = lambda point : point[0])
                current_max_gini = 0
                current_value_maximizing = 0
                gini_start = self.get_gini()
                label_counter = {'left_false': 0, 'left_true': 0, 'right_false': 0, 'right_true': 0}
                true_number = sum(self.labels)
                label_counter['right_true'] = true_number
                label_counter['right_false'] = n - true_number
                threshold = 1
                while threshold < n: #We don't need to check when one of the sets is empty   
                    current_label = continuous_values[threshold-1][1]
                    if current_label:
                        label_counter['left_true'] += 1
                        label_counter['right_true'] -= 1
                    else:
                        label_counter['left_false'] += 1
                        label_counter['right_false'] -= 1
                    
                    #If two values are identical, we cannot store them in differents groups, because however the split threshold would not be respected
                    if continuous_values[threshold-1][0] == continuous_values[threshold][0]:
                        threshold += 1
                        continue                    
                    
                    if threshold < self.min_split_points or n - threshold < self.min_split_points:
                        threshold += 1
                        continue
                    
                    n1 = threshold
                    n2 = n - threshold
                    
                    gini1 = 1 - ((label_counter['left_false'])**2 + (label_counter['left_true'])**2)/((label_counter['left_false']+label_counter['left_true'])**2)
                    gini2 = 1 - ((label_counter['right_false'])**2 + (label_counter['right_true'])**2)/((label_counter['right_false']+label_counter['right_true'])**2)
                    gini_split = (n1/n)*gini1 + (n2/n)*gini2

                    gini_gain = gini_start - gini_split
                    if gini_gain > current_max_gini:
                        current_value_maximizing = (continuous_values[threshold-1][0] + continuous_values[threshold][0])/2
                        current_max_gini = gini_gain
                    threshold += 1
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

    def get_gini_2(self, values):
        label_frequency = [0,0]
        for label in values:
            if label[1] == 0:
                label_frequency[0] += 1
            else:
                label_frequency[1] += 1
        point_count = len(values)
        label_frequency = list(map(lambda x:x/point_count, label_frequency))
        return 1 - label_frequency[0]**2 - label_frequency[1]**2