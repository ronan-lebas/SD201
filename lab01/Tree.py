from typing import List

from PointSet import PointSet, FeaturesTypes

import numpy as np

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1,
                 beta: float = 0
                 ):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.parent = None
        self.points = PointSet(features, labels, types)
        self.points.min_split_points = min_split_points
        self.types = types
        self.h = h
        self.min_split_points = min_split_points
        self.beta = beta
        self.update_counter = 0
        if self.h == 0:
            self.guess_label = self.most_frequent_label(labels)
            return
        
        if labels == len(labels)*True:
            self.guess_label = True
            self.h = 0
            return
        if labels == len(labels)*False:
            self.guess_label = False
            self.h = 0
            return
        
        else:    
            feature_index = self.points.get_best_gain()[0]
            self.deciding_feature = feature_index
            child1_features = []
            child2_features = []
            child1_labels = []
            child2_labels = []
            
            if self.types[feature_index] == FeaturesTypes.BOOLEAN:
                for i,f in enumerate(features):
                    if f[feature_index] == 0:
                        child1_features.append(f)
                        child1_labels.append(labels[i])
                    else:
                        child2_features.append(f)
                        child2_labels.append(labels[i])
                if len(child1_features) < min_split_points:
                    self.guess_label = self.most_frequent_label(child2_labels)
                    self.h = 0
                    return
                if len(child2_features) < min_split_points:
                    self.guess_label = self.most_frequent_label(child1_labels)
                    self.h = 0
                    return
                self.child1 = Tree(child1_features, child1_labels, types, self.h-1, min_split_points, self.beta)
                self.child2 = Tree(child2_features, child2_labels, types, self.h-1, min_split_points, self.beta)

            elif self.types[feature_index] == FeaturesTypes.CLASSES:
                for i,f in enumerate(features):
                    if f[feature_index] == self.points.maximizing_feature[feature_index]:
                        child1_features.append(f)
                        child1_labels.append(labels[i])
                    else:
                        child2_features.append(f)
                        child2_labels.append(labels[i])
                if len(child1_features) < min_split_points:
                    self.guess_label = self.most_frequent_label(child2_labels)
                    self.h = 0
                    return
                if len(child2_features) < min_split_points:
                    self.guess_label = self.most_frequent_label(child1_labels)
                    self.h = 0
                    return
                self.child1 = Tree(child1_features, child1_labels, types, self.h-1, min_split_points, self.beta)
                self.child2 = Tree(child2_features, child2_labels, types, self.h-1, min_split_points, self.beta)

            elif self.types[feature_index] == FeaturesTypes.REAL:
                for i,f in enumerate(features):
                    if f[feature_index] < self.points.maximizing_feature[feature_index]:
                        child1_features.append(f)
                        child1_labels.append(labels[i])
                    else:
                        child2_features.append(f)
                        child2_labels.append(labels[i])
                if len(child1_features) < min_split_points:
                    self.guess_label = self.most_frequent_label(child2_labels)
                    self.h = 0
                    return
                if len(child2_features) < min_split_points:
                    self.guess_label = self.most_frequent_label(child1_labels)
                    self.h = 0
                    return
                self.child1 = Tree(child1_features, child1_labels, types, self.h-1, min_split_points, self.beta)
                self.child2 = Tree(child2_features, child2_labels, types, self.h-1, min_split_points, self.beta)
            self.child1.parent = self
            self.child2.parent = self
            self.child1.is_left_child = True
            self.child2.is_left_child = False

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        if self.h == 0:
            return self.guess_label        
        else:
            if self.types[self.deciding_feature] == FeaturesTypes.BOOLEAN:
                return self.child1.decide(features) if not features[self.deciding_feature] else self.child2.decide(features)
            elif self.types[self.deciding_feature] == FeaturesTypes.CLASSES:
                return self.child1.decide(features) if features[self.deciding_feature] == self.points.maximizing_feature[self.deciding_feature] else self.child2.decide(features)
            elif self.types[self.deciding_feature] == FeaturesTypes.REAL:
                return self.child1.decide(features) if features[self.deciding_feature] < self.points.maximizing_feature[self.deciding_feature] else self.child2.decide(features)

    def most_frequent_label(self,labels: List[bool]) -> bool:
        """Give the most frequent label of a list

        Parameters
        ----------
            labels : List[bool]
                A list of labels

        Returns
        -------
            bool
                The most frequent label
        """
        c1, c2 = 0, 0
        for i in range(len(labels)):
            if not labels[i]: c1 += 1
            else: c2 += 1
        return 0 if c1>c2 else 1

    #For the following two functions : the only problem seems to be a memory problem (references, pointers etc.)

    def add_training_point(self, features: List[float], label: bool) -> None:
        self.update_counter += 1
        npfeatures = np.asarray(features).reshape(1,self.points.features.shape[1])
        self.points.features = np.concatenate((self.points.features, npfeatures), axis=0)
        self.points.labels.append(label)
        #If the update counter exceeds the threshold, reconstruct the subtree
        if self.update_counter >= self.beta*len(self.points.labels):
            temp = Tree(self.points.features, self.points.labels, self.types, self.h, self.min_split_points, self.beta)    
            if self.parent != None and self.is_left_child:
                self.parent.child1 = temp
            elif self.parent != None:
                self.parent.child2 = temp
        #Propagation of the new point
        elif self.h > 0:
            if self.types[self.deciding_feature] == FeaturesTypes.BOOLEAN:
                return self.child1.add_training_point(features, label) if not features[self.deciding_feature] else self.child2.add_training_point(features, label)
            elif self.types[self.deciding_feature] == FeaturesTypes.CLASSES:
                return self.child1.add_training_point(features, label) if features[self.deciding_feature] == self.points.maximizing_feature[self.deciding_feature] else self.child2.add_training_point(features, label)
            elif self.types[self.deciding_feature] == FeaturesTypes.REAL:
                return self.child1.add_training_point(features, label) if features[self.deciding_feature] < self.points.maximizing_feature[self.deciding_feature] else self.child2.add_training_point(features, label)
        
    def del_training_point(self, features: List[float], label: bool) -> None:
        self.update_counter += 1
        npfeatures = np.asarray(features).reshape(1, self.points.features.shape[1])
        #First, propagate before removing if this is not a leaf and if the removing won't trigger an update
        #We go along the tree, like in decide, but to retrieve the point to remove
        if self.h > 0 and self.update_counter < self.beta*(len(self.points.features)-1):
            if self.types[self.deciding_feature] == FeaturesTypes.BOOLEAN:
                self.child1.del_training_point(features, label) if not features[self.deciding_feature] else self.child2.del_training_point(features, label)
            elif self.types[self.deciding_feature] == FeaturesTypes.CLASSES:
                self.child1.del_training_point(features, label) if features[self.deciding_feature] == self.points.maximizing_feature[self.deciding_feature] else self.child2.del_training_point(features, label)
            elif self.types[self.deciding_feature] == FeaturesTypes.REAL:
                self.child1.del_training_point(features, label) if features[self.deciding_feature] < self.points.maximizing_feature[self.deciding_feature] else self.child2.del_training_point(features, label)
        #Removal of the point
        for i in range(len(self.points.features)):
            if np.array_equal(self.points.features[i], npfeatures) and self.points.labels[i] == label:
                np.delete(self.points.features, i, axis=0)
                self.points.labels.pop(i)
                break
        #If the update counter exceeds the threshold, reconstruct the subtree
        if self.update_counter >= self.beta*len(self.points.features):
            temp = Tree(self.points.features, self.points.labels, self.types, self.h, self.min_split_points, self.beta)    
            if self.parent != None and self.is_left_child:
                self.parent.child1 = temp
            elif self.parent != None:
                self.parent.child2 = temp