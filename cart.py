'''TODO'''
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


class TreeNode:
    '''Tree for dealing with Regression and Classification tasks in a partitioned fashion'''

    @classmethod
    def bin_split_dataset(cls, dataset: np.ndarray, feature: int,
                        threshold: float):
        '''Split the dataset with respect to a feature and threshold'''
        gt_threshold = dataset[:, feature] > threshold
        leq_threshold = dataset[:, feature] <= threshold
        left_set = dataset[gt_threshold.nonzero(), :][0]
        right_set = dataset[leq_threshold.nonzero(), :][0]
        return left_set, right_set

    @classmethod
    def choose_best_split(cls, dataset: np.ndarray, leaf_type: Callable,
                        err_type: Callable, tols: tuple):
        '''Choose the best split value'''
        tolerance_s, tolerance_n = tols
        deviation = err_type(dataset)
        if np.unique(dataset[:, -1], axis=0).shape[0] == 1:
            return None, leaf_type(dataset), deviation
        columns = dataset.shape[1]
        best_deviation, best_index, best_value = np.inf, 0, 0
        for feature_index in range(columns-1):
            for split_value in set(dataset[:, feature_index]):
                left_set, right_set = TreeNode.bin_split_dataset(
                    dataset, feature_index, split_value)
                if not ((left_set.shape[0] < tolerance_n)
                        or (right_set.shape[0] < tolerance_n)):
                    new_deviation = err_type(left_set)+err_type(right_set)
                    if new_deviation < best_deviation:
                        best_index = feature_index
                        best_value = split_value
                        best_deviation = new_deviation
        if (deviation - best_deviation) < tolerance_s:
            return None, leaf_type(dataset), best_deviation
        left_set, right_set = TreeNode.bin_split_dataset(
            dataset, best_index, best_value)
        if (left_set.shape[0] < tolerance_n) or (right_set.shape[0] < tolerance_n):
            return None, leaf_type(dataset), best_deviation
        return best_index, best_value, best_deviation

    def __init__(self, input_data: np.ndarray, output: np.ndarray,
                leaf_type: Callable, err_type: Callable, tols: tuple = (1, 4)):
        if input_data.shape[0] != output.shape[0]:
            raise Exception(
                "inputs and outputs shall have the same number of rows")
        dataset = np.c_[input_data, output]
        self.feature, self.threshold, self.deviation = TreeNode.choose_best_split(
            dataset, leaf_type, err_type, tols)
        if self.feature is None:
            self.left = None
            self.right = None
        else:
            left_set, right_set = TreeNode.bin_split_dataset(
                dataset, self.feature, self.threshold)
            self.left = TreeNode(
                left_set[:, :-1], left_set[:, -1], leaf_type, err_type, tols)
            self.right = TreeNode(
                right_set[:, :-1], right_set[:, -1], leaf_type, err_type, tols)

    def display_tree(self, children: list = None, ind: int = 1):
        '''Display the tree architecture'''
        if children is None:
            print(' '*ind, round(self.threshold,2),' ', self.feature)
            self.display_tree([self.left, self.right])
        elif len(children):
            new_children = []
            for child in children:
                if child:
                    print(' '*ind, round(child.threshold,2),' ', child.feature, end='')
                    if child.left:
                        new_children.append(child.left)
                    if child.right:
                        new_children.append(child.right)
            print('')
            self.display_tree(new_children, ind+1)

    def is_leaf(self):
        '''Verify if the object is a TreeNode instance'''
        return self.left is None and self.right is None

    def get_height(self) -> int:
        '''Get the height of the tree'''
        left_height = 0 if self.left.is_leaf() else self.left.get_height()
        right_height = 0 if self.right.is_leaf() else self.right.get_height()
        return max(left_height, right_height)+1

    def get_leaves_deviation(self) -> float:
        '''Get the total leaves deviation of the tree'''
        if self.is_leaf():
            return self.deviation
        left_leaves = self.left.get_leaves_deviation()
        right_leaves = self.right.get_leaves_deviation()
        return left_leaves+right_leaves

    def get_number_of_leaves(self) -> int:
        '''Get the number of leaves of the tree'''
        if self.is_leaf():
            return 1
        left_leaves = self.left.get_number_of_leaves()
        right_leaves = self.right.get_number_of_leaves()
        return left_leaves+right_leaves

    def get_deviation_by_number_of_leaves(self):
        '''Get the total leaves deviation by the number of leaves of the tree'''
        return self.get_leaves_deviation()/float(self.get_number_of_leaves())

    def predict(self, input_data: np.ndarray):
        '''Predict the output for the input_data'''
        if not self.is_leaf():
            if input_data[self.feature] <= self.threshold:
                return self.right.predict(input_data)
            return self.left.predict(input_data)
        return self.threshold

def standard_err(dataset: np.ndarray) -> float:
    '''Get the variance multiplied by the total quantity of the labeled output'''
    output = dataset[:, -1]
    return output.var()*output.shape[0]

def standard_leaf(dataset: np.ndarray) -> float:
    '''Get the mean of the labeled output'''
    output = dataset[:, -1]
    return output.mean()

def linear_solve(dataset: np.ndarray):
    '''Solve linear system'''
    rows, columns = dataset.shape
    input_data = np.ones((rows, columns))
    input_data[:, 1:], output = dataset[:, :-1], dataset[:,-1]
    try:
        weights = np.linalg.inv(input_data.T@input_data)@(input_data.T@output)
        return weights, input_data, output
    except np.linalg.LinAlgError:
        print("This matrix is singular, cannot do inverse")
        return np.array([]), input_data, output

def model_leaf(dataset: np.ndarray):
    '''Get the weights associated with the linear solve'''
    weights, _, _ = linear_solve(dataset)
    return weights

def model_err(dataset: np.ndarray):
    '''Get the deviance of the estimated output'''
    weights, input_data, output = linear_solve(dataset)
    estimated_output = input_data @ weights
    return np.sum(np.power(output-estimated_output, 2))

def leaked_relu(variable: float):
    '''Leaked ReLU function'''
    if variable > 0.3:
        return 12*variable
    return 3.3+variable

vectorized_leaked_relu = np.vectorize(leaked_relu)

def get_line(weights: list[float], variable: float):
    '''Constructs a linear function'''
    return weights[1]*variable+weights[0]

if __name__ == "__main__":
    xx = np.linspace(0,100,10000)
    yy = np.sin(xx) + np.random.normal(0,0.1,10000)
    tree_constant = TreeNode(xx.T, yy, standard_leaf, standard_err, (3,10))
    tree_linear = TreeNode(xx.T, yy, model_leaf, model_err, (3,10))
    yy_constant = [tree_constant.predict(np.array([x])) for x in xx]
    yy_linear = [get_line(tree_linear.predict(np.array([x])), x) for x in xx]
    plt.scatter(xx, yy)
    plt.plot(xx, yy_constant, color='purple')
    plt.plot(xx, yy_linear, color='orange')
    plt.show()
