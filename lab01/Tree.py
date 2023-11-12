from typing import List

from PointSet import PointSet, FeaturesTypes

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
                 min_split_points: int = 1):
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
        self.points = PointSet(features, labels, types)
        self.points.min_split_points = min_split_points
        self.types = types
        self.h = h
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
                self.child1 = Tree(child1_features, child1_labels, types, self.h-1, min_split_points)
                self.child2 = Tree(child2_features, child2_labels, types, self.h-1, min_split_points)

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
                self.child1 = Tree(child1_features, child1_labels, types, self.h-1, min_split_points)
                self.child2 = Tree(child2_features, child2_labels, types, self.h-1, min_split_points)

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
                self.child1 = Tree(child1_features, child1_labels, types, self.h-1, min_split_points)
                self.child2 = Tree(child2_features, child2_labels, types, self.h-1, min_split_points)


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