# import copy
# import uuid
# import pickle
import pprint
import collections
from collections import defaultdict
from math import log2
import pandas as pd
'''Wenjie Shi & Yidan Xue
'''
class DecisionTree(object):
    ''' use ID3 as dataset classifier
    '''
    def __init__(self):
        self.isDecision = False
        self.branches = []

    @staticmethod
    def split_dataset(dataset, classes, feat_idx):
        ''' classify dataset according to certain feature
        :param dataset: dataset to be splited, vector list
        :param classes: dataset classes,length=len(dataset)
        :param feat_idx: feature index in feature vector
        :param splited_dict: save splitted data's dict feature value: [subdataset, subclass_list]
        '''
        splited_dict = {}
        for data_vect, cls in zip(dataset, classes):
            feat_val = data_vect[feat_idx]
            sub_dataset, sub_classes = splited_dict.setdefault(feat_val, [[], []])
            sub_dataset.append(data_vect[: feat_idx] + data_vect[feat_idx+1: ])
            sub_classes.append(cls)

        return splited_dict

    def get_shanno_entropy(self, values):
        ''' calculate Shanno Entropy,input is a list
        '''
        uniq_vals = set(values)
        val_nums = {key: values.count(key) for key in uniq_vals}
        probs = [v/len(values) for k, v in val_nums.items()]
        entropy = sum([-prob*log2(prob) for prob in probs])
        return entropy

    def choose_best_split_feature(self, dataset, classes):
        ''' choose best split feature according to information gain
        :param dataset: dataset to be splitted
        :param classes: dataset classes
        :return: entropy_gains.index(max(entropy_gains))
        '''
        base_entropy = self.get_shanno_entropy(classes)

        feat_num = len(dataset[0])
        entropy_gains = []
        for i in range(feat_num):
            splited_dict = self.split_dataset(dataset, classes, i)
            new_entropy = sum([
                len(sub_classes)/len(classes)*self.get_shanno_entropy(sub_classes)
                for _, (_, sub_classes) in splited_dict.items()
            ])
            entropy_gains.append(base_entropy - new_entropy)

        return entropy_gains.index(max(entropy_gains))


    def create_tree(self, dataset, classes, feat_names):
        ''' create decision tree recursively based on current dataset
        :param dataset: dataset
        :param feat_names: feature names of data in this dataset
        :param classes: data's classes
        :param tree: return decision tree as dictionary
        '''
        # stop while classes=1 in current dataset
        if len(set(classes)) == 1:

            return classes[0]

        # after iterate all features,return the one has the majority
        if len(feat_names) == 0:

            return get_majority(classes)

        # split newly created subtrees
        self.isDecision=True
        tree = {}
        best_feat_idx = self.choose_best_split_feature(dataset, classes)
        feature = feat_names[best_feat_idx]
        tree[feature] = {}

        # create subdataset for creating subtrees
        sub_feat_names = feat_names[:]
        sub_feat_names.pop(best_feat_idx)

        splited_dict = self.split_dataset(dataset, classes, best_feat_idx)
        for feat_val, (sub_dataset, sub_classes) in splited_dict.items():
            tree[feature][feat_val] = self.create_tree(sub_dataset,
                                                       sub_classes,
                                                       sub_feat_names)
        self.tree = tree
        self.feat_names = feat_names

        return tree

    def get_nodes_edges(self, tree=None, root_node=None):
        ''' retrun all nodes and edges in the tree
        '''
        t1=('Node', ['id', 'label'])
        t2=('Edge', ['start', 'end', 'label'])
        Node = t1
        Edge = t2

        if tree is None:
            tree = self.tree

        if type(tree) is not dict:
            return [], []

        nodes, edges = [], []

        if root_node is None:
            label = list(tree.keys())[0]
            root_node = Node._make([uuid.uuid4(), label])
            nodes.append(root_node)

        for edge_label, sub_tree in tree[root_node.label].items():
            node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
            sub_node = Node._make([uuid.uuid4(), node_label])
            nodes.append(sub_node)

            edge = Edge._make([root_node, sub_node, edge_label])
            edges.append(edge)

            sub_nodes, sub_edges = self.get_nodes_edges(sub_tree, root_node=sub_node)
            nodes.extend(sub_nodes)
            edges.extend(sub_edges)

        return nodes, edges

    def dotify(self, tree=None):
        ''' get Graphviz Dot flie content of tree
        '''
        if tree is None:
            tree = self.tree

        content = 'digraph decision_tree {\n'
        nodes, edges = self.get_nodes_edges(tree)

        for node in nodes:
            content += '    "{}" [label="{}"];\n'.format(node.id, node.label)

        for edge in edges:
            start, label, end = edge.start, edge.label, edge.end
            content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label)
        content += '}'

        return content


if __name__=="__main__":
    with open('dt-data.txt', 'r') as f:
        line = f.readline().replace('(','').replace(')\n','')
        labels=line.split(', ')

        line = f.readline()
        line = f.readline()
        dataSet=[]
        while line:
            line.replace('\n','')
            tmp=line.split(': ')
            line_list=tmp[1].replace(";\n",'').split(', ')
            dataSet.append(line_list[0:7])
            line=f.readline()
    def get_majority(classes):
        ''' return class that gets majority
        '''
        cls_num = defaultdict(lambda: 0)
        for cls in classes:
            cls_num[cls] += 1

        return max(cls_num, key=cls_num.get)
    data_set = [item[0:6] for item in dataSet]
    feat_names = labels[0:6]
    classes = [item[6] for item in dataSet]
    c=DecisionTree()
    result=c.create_tree(data_set,classes,feat_names)
    def pretty(d, indent=0):
        for key, value in d.items():

            print('  ' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('  ' * (indent+1) + str(value))
    pretty(result)
    test={'Occupied':'Moderate','Price':'Cheap','Music':'Loud','Location':'City-Center','VIP':'No','Favorite Beer':'No'}
    print("Prediction for:",test)
    while result not in ('Yes','No'):
            k=list(result.keys())
            k=k[0]
            v=list(result.values())
            v=v[0]
            c1=test.get(k)
            result=v.get(c1)
    print("Prediction result:",result)
