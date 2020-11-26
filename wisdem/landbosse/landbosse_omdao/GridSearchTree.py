import numpy as np
import pandas as pd

"""
This module contains the logic to handle a tree to compute
points in an N-dimensional parametric search space.
"""


class GridSearchTreeNode:
    """
    This just contains information about a node in the grid
    search tree.
    """

    def __init__(self):
        self.cell_specification = None
        self.children = []
        self.value = None


class GridSearchTree:
    """
    This class implements a k-ary tree to compute possible
    combinations of points in a N-dimensional parametric
    search space.
    """

    def __init__(self, parametric_list):
        """
        This simply sets the parametric_list. See the first dataframe
        described in the docstring of XlsxReader.create_parametric_value_list()

        Parameters
        ----------
        parametric_list : pandas.DataFrame
            The dataframe of the parametrics list.
        """
        self.parametric_list = parametric_list

    def build_grid_tree_and_return_grid(self):
        """
        See the dataframes in XlsxReader.create_parametric_value_list()
        for context.

        This builds a tree of points in the search space and traverse
        it to find points on the grid.

        Returns
        -------
        """

        # Build the tree. Its leaf nodes contain the values for each
        # point in the grid.
        root = self.build_tree()

        # Recursions of the traversal method needs to start with an empty
        # list.
        grid = self.dfs_search_tree(root, traversal=[])
        return grid

    def build_tree(self, depth=0, root=None):
        """
        This method builds a k-ary tree to contain cell_specifications and
        their values.

        Callers from outside this method shouldn't override the defaults
        for the parameters. These parameters are to manage the recursion,
        and are supplied by this method when it invokes itself.

        Parameters
        ----------
        root : GridSearchTreeNode
            The root of the subtree. At the start of iteration, at the
            root of the whole tree, this should be None.

        depth : int
            The level of the tree currently being built. This is
            also the row number in the dataframe from which the tree
            is being built.

        Returns
        -------
        GridSearchTreeNode
            The root of the tree just built.
        """
        row = self.parametric_list.iloc[depth]
        cell_specification = f"{row['Dataframe name']}/{row['Row name']}/{row['Column name']}"

        # First, make an iterable of the range we are going to be using.
        if "Value list" in row and not pd.isnull(row["Value list"]):
            values = [float(value) for value in row["Value list"].split(",")]
        else:
            start = row["Min"]
            end = row["Max"]
            step = row["Step"]
            values = np.arange(start, end + step, step)

        if root == None:
            root = GridSearchTreeNode()

        # Putting the stop at end + step ensures the end value is in the sequence
        #
        # Append children for each value in the parametric step sequence.

        for value in values:
            child = GridSearchTreeNode()
            child.value = value
            child.cell_specification = cell_specification
            root.children.append(child)

            # If there are more levels of variables to add, recurse
            # down 1 level.
            if len(self.parametric_list) > depth + 1:
                self.build_tree(depth + 1, child)

        return root

    def dfs_search_tree(self, root, traversal, path=None):
        """
        This does a depth first search traversal of the GridSearchTree
        specified by the root parameter. It stores the node it encounters
        in the list referenced by traversal.

        There is a distinction from normal DFS traversals: Only leaf nodes
        are recorded in the traversal. This means that only nodes that have
        a complete list of cell specifications and values are returned.

        Parameters
        ----------
        root : GridSearchTreeNode
            The root of the

        traversal : list
            The nodes traversed on the tree. When this method is called
            by an external caller, this should be an empty list ([])

        path : list
            This shouldn't be manipulated except by this method itself.
            It is for storing the paths to the leaf nodes.

        Returns
        -------
        list
            A list of dictionaries that hold the cell specifications and
            values of each leaf node.
        """

        path = [] if path is None else path[:]

        if root.cell_specification is not None:
            path.append(
                {
                    "cell_specification": root.cell_specification,
                    "value": root.value,
                }
            )

        if len(root.children) == 0:
            traversal.append(path)

        for child in root.children:
            self.dfs_search_tree(child, traversal, path)

        return traversal
