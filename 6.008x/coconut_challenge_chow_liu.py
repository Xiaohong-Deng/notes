import numpy as np


def get_labeled_windowed_data(observations, window_size=7):
    """
    Split up the observations into windowed chunks. Each windowed chunk of
    observations is associated with a label vector of what the price change is
    per market *immediately after* the windowed chunk (+1 for price goes up,
    0 for no change, and -1 for price goes down). Thus, a classifier's task
    for the data is given a windowed chunk, to predict what its label is
    (i.e., given recent percent changes in all the markets, predict the
    directions of the next price changes per market).

    Inputs
    ------
    - observations: 2D array; each column is a percent-change time series data
        for a specific market
    - window_size: how large the window is (in number of time points)

    Outputs
    -------
    - windows: 3D array; each element of the outermost array is a 2D array
        of the same format as `observations` except where the number of time
        points is exactly `window_size`
    - window_labels: 2D array; `window_labels[i]` is a 1D vector of labels
        corresponding to the time point *after* the window specified by
        `windows[i]`; `window_labels[i]` says what the price change is for
        each market (+1 for going up, 0 for staying the same, and -1 for going
        down)

    *WARNING*: Note that the training data produced here is inherently not
    i.i.d. in that `windows[0]` and `windows[1]`, for instance, will largely
    overlap!
    """
    num_time_points, num_markets = observations.shape
    windows = []
    window_labels = []
    for start_idx in range(num_time_points - window_size):
        windows.append(observations[start_idx:start_idx + window_size + 1])
        # the sign of the 8th day's movement, +1 for up, -1 for down
        # 0 for still
        window_labels.append(1 * (observations[start_idx + window_size] > 0)
                             - 1 * (observations[start_idx + window_size] < 0))
    # each window includes 8 days data, i.e, 32 price movements
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    return windows, window_labels


def convert_tree_as_set_to_adjacencies(tree):
    """
    This snippet of code converts between two representations we use for
    edges (namely, with Chow-Liu it suffices to just store edges as a set of
    pairs (i, j) with i < j), whereas when we deal with learning tree
    parameters and code Sum-Product it will be convenient to have an
    "adjacency list" representation, where we can query and find out the list
    of neighbors for any node. We store this "adjacency list" as a Python
    dictionary.

    Input
    -----
    - tree: a Python set of edges (where (i, j) being in the set means that we
        don't have to have (j, i) also stored in this set)

    Output
    ------
    - edges: a Python dictionary where `edges[i]` gives you a list of neighbors
        of node `i`
    """
    edges = {}
    for i, j in tree:
        if i not in edges:
            edges[i] = [j]
        else:
            edges[i].append(j)
        if j not in edges:
            edges[j] = [i]
        else:
            edges[j].append(i)
    return edges


class UnionFind():

    def __init__(self, nodes):
        """
        Union-Find data structure initialization sets each node to be its own
        parent (so that each node is in its own set/connected component), and
        to also have rank 0.

        Input
        -----
        - nodes: list of nodes
        """
        self.parents = {}
        self.ranks = {}

        for node in nodes:
            self.parents[node] = node
            self.ranks[node] = 0

    def find(self, node):
        """
        Finds which set/connected component that a node belongs to by returning
        the root node within that set.

        Technical remark: The code here implements path compression.

        Input
        -----
        - node: the node that we want to figure out which set/connected
            component it belongs to

        Output
        ------
        the root node for the set/connected component that `node` is in
        """
        if self.parents[node] != node:
            # path compression
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]

    def union(self, node1, node2):
        """
        Merges the connected components of two nodes.

        Inputs
        ------
        - node1: first node
        - node2: second node
        """
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:  # only merge if the connected components differ
            if self.ranks[root1] > self.ranks[root2]:
                self.parents[root2] = root1
            else:
                self.parents[root1] = root2
                if self.ranks[root1] == self.ranks[root2]:
                    self.ranks[root2] += 1


def compute_empirical_distribution(values):
    """
    Given a sequence of values, compute the empirical distribution.

    Input
    -----
    - values: list (or 1D NumPy array or some other iterable) of values

    Output
    ------
    - distribution: a Python dictionary representing the empirical distribution
    """
    distribution = {}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    size = len(values)

    for v in values:
        if distribution.get(v) is None:
            distribution[v] = 1
        else:
            distribution[v] += 1

    for k in distribution:
        distribution[k] /= size
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return distribution


def compute_empirical_mutual_info_nats(var1_values, var2_values):
    """
    Compute the empirical mutual information for two random variables given a
    pair of observed sequences of those two random variables.

    Inputs
    ------
    - var1_values: observed sequence of values for the first random variable
    - var2_values: observed sequence of values for the second random variable
        where it is assumed that the i-th entries of `var1_values` and
        `var2_values` co-occur

    Output
    ------
    The empirical mutual information *in nats* (not bits)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #

    empirical_mutual_info_nats = 0.0

    size = len(var1_values)
    var1_distribution = compute_empirical_distribution(var1_values)
    var2_distribution = compute_empirical_distribution(var2_values)
    joint_var1_var2_distribution = {}

    for (i, j) in zip(var1_values, var2_values):
        if joint_var1_var2_distribution.get((i, j)) is None:
            joint_var1_var2_distribution[(i, j)] = 1
        else:
            joint_var1_var2_distribution[(i, j)] += 1

    for k in joint_var1_var2_distribution:
        joint_var1_var2_distribution[k] /= size

    var1_alphabet = set(var1_values)
    var2_alphabet = set(var2_values)

    for i in var1_alphabet:
        for j in var2_alphabet:
            joint_x_y = joint_var1_var2_distribution[(i, j)]
            empirical_mutual_info_nats += joint_x_y * \
                np.log(joint_x_y /
                       (var1_distribution[i] * var2_distribution[j]))
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return empirical_mutual_info_nats


def chow_liu(flattened_windows):
    """
    Run the chow-liu to learn the tree
    Inputs
    ----
    - flattened_windows: before passing the windows
      to chow_liu we flatten the 2D window such that each column
      corresponds to a var, each line corresponds to a window
      it's easier for mutual info computing
    """
    best_tree = set()  # we will add in edges to this set
    num_obs, num_vars = flattened_windows.shape
    union_find = UnionFind(range(num_vars))
    mutual_info = {}
    for var1 in range(num_vars):
        for var2 in range(var1 + 1, num_vars):
            edge_weight = compute_empirical_mutual_info_nats(
                flattened_windows[:, var1], flattened_windows[:, var2])
            mutual_info[(var1, var2)] = edge_weight

    while len(mutual_info) != 0:
        edge_to_add = max(mutual_info, key=mutual_info.get)
        if union_find.find(edge_to_add[0]) != union_find.find(edge_to_add[1]):
            union_find.union(edge_to_add[0], edge_to_add[1])
            best_tree.add(edge_to_add)
        mutual_info.pop(edge_to_add)
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return best_tree


def compute_empirical_conditional_distribution(var1_values, var2_values):
    """
    Given two sequences of values (corresponding to samples from two
    random variables), compute the empirical conditional distribution of
    the first variable conditioned on the second variable.

    Inputs
    ------
    - var1_values: list (or 1D NumPy array or some other iterable) of values
        sampled from, say, $X_1$
    - var2_values: list (or 1D NumPy array or some other iterable) of values
        sampled from, say, $X_2$, where it is assumed that the i-th entries of
        `var1_values` and `var2_values` co-occur
    Output
    ------
    - conditional_distributions: a dictionary consisting of dictionaries;
        `conditional_distributions[x_2]` should be the dictionary that
        represents the conditional distribution $X_1$ given $X_2 = x_2$
    """
    conditional_distributions = {x2: {} for x2 in set(var2_values)}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    conditional_sample_size = {x2: 0 for x2 in set(var2_values)}

    for idx in range(len(var1_values)):
        x2 = var2_values[idx]
        x1 = var1_values[idx]
        conditional_sample_size[x2] += 1
        if conditional_distributions[x2].get(x1) is None:
            conditional_distributions[x2][x1] = 1
        else:
            conditional_distributions[x2][x1] += 1

    for x2 in conditional_distributions:
        for x1 in conditional_distributions[x2]:
            conditional_distributions[x2][x1] /= conditional_sample_size[x2]

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return conditional_distributions


def learn_tree_parameters(observations, tree, root_node=0):
    """
    Learn a collection of node and edge potentials from observations that
    corresponds to a maximum likelihood estimate.

    Please use the approach presented in the course video/notes. Remember that
    the only node potential that isn't all 1's is the one corresponding to the
    root node chosen, and the edge potentials are set to be empirical
    conditional probability distributions.

    Inputs
    ------
    - observations: a 2D NumPy array where the i-th row corresponds to the
        i-th training data point

        *IMPORTANT*: it is assumed that the nodes in the graphical model are
        numbered 0, 1, ..., up to the number of variables minus 1, where the
        number of variables in the graph is determined from `observations` by
        looking at `observations.shape[1]`
    - tree: a set consisting of which edges are present (if (i, j) is in the
        set, then you don't have to also include (j, i)); note that the
        nodes must be as stated above
    - root_node: an integer specifying which node to treat as the root node

    Outputs
    -------
    - node_potentials: Python dictionary where `node_potentials[i]` is
        another Python dictionary representing the node potential table for
        node `i`; this means that `node_potentials[i][x_i]` should give the
        potential value for what, in the course notes, we call $\phi_i(x_i)$
    - edge_potentials: Python dictionary where `edge_potentials[(i, j)]` is
        a dictionaries-within-a-dictionary representation for a 2D potential
        table so that `edge_potentials[(i, j)][x_i][x_j]` corresponds to
        what, in the course notes, we call $\psi_{i,j}(x_i, x_j)$

        *IMPORTANT*: For the purposes of this project, please be sure to
        specify both `edge_potentials[(i, j)]` *and* `edge_potentials[(j, i)]`,
        where `edge_potentials[(i, j)][x_i][x_j]` should equal
        `edge_potentials[(j, i)][x_j][x_i]` -- we have provided a helper
        function `transpose_2d_table` below that, given edge potentials
        computed in one "direction" (i, j), computes the edge potential
        for the "other direction" (j, i)
    """
    nodes = set(range(observations.shape[1]))
    edges = convert_tree_as_set_to_adjacencies(tree)
    node_potentials = {}
    edge_potentials = {}

    def transpose_2d_table(dicts_within_dict_table):
        """
        Given a dictionaries-within-dictionary representation of a 2D table
        `dicts_within_dict_table`, computes a new 2D table that's also a
        dictionaries-within-dictionary representation that is the transpose of
        the original 2D table, so that:

            transposed_table[x1][x2] = dicts_within_dict_table[x2][x1]

        Input
        -----
        - dicts_within_dict_table: as described above

        Output
        ------
        - transposed_table: as described above
        """
        transposed_table = {}
        for x2 in dicts_within_dict_table:
            for x1 in dicts_within_dict_table[x2]:
                if x1 not in transposed_table:
                    transposed_table[x1] = \
                        {x2: dicts_within_dict_table[x2][x1]}
                else:
                    transposed_table[x1][x2] = \
                        dicts_within_dict_table[x2][x1]
        return transposed_table

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    node_potentials[root_node] = compute_empirical_distribution(observations[
                                                                :, root_node])
    fringe = [root_node]
    visited = {node: False for node in nodes}
    while len(fringe) > 0:
        node = fringe.pop(0)
        visited[node] = True
        for neighbor in edges[node]:
            if not visited[neighbor]:
                node_obs = observations[:, node]
                neighbor_obs = observations[:, neighbor]
                conditional_distributions = compute_empirical_conditional_distribution(
                    neighbor_obs, node_obs)
                edge_potentials[(node, neighbor)] = conditional_distributions
                edge_potentials[(neighbor, node)] = transpose_2d_table(
                    conditional_distributions)
                fringe.append(neighbor)

    # what if some vals are missing, what if only 1 out of
    # 3 possible vals showed in empirical data?
    nodes.remove(root_node)
    for node in nodes:
        node_potentials[node] = {}
        for val in set(observations[:, node]):
            node_potentials[node][val] = 1
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return node_potentials, edge_potentials


def sum_product(nodes, edges, node_potentials, edge_potentials):
    """
    Run the Sum-Product algorithm.

    Inputs
    ------
    - nodes: Python set that consists of the nodes
    - edges: Python dictionary where `edges[i]` is a list saying which nodes
        are neighbors of node `i`
    - node_potentials: Python dictionary where `node_potentials[i]` is
        another Python dictionary representing the node potential table for
        node `i`; this means that `node_potentials[i][x_i]` should give the
        potential value for what, in the course notes, we call $\phi_i(x_i)$

        *IMPORTANT*: For the purposes of this project, the alphabets of each
        random variable should be inferred from the node potentials, so each
        node potential's dictionary's keys should tell you what the alphabet is
        (or at least the subset of the alphabet for which the probability is
        nonzero); this means that you should not use collections.defaultdict
        to produce, for instance, a dictionary with no keys that outputs 1 for
        everything here since we cannot read off what the alphabet is for the
        random variable
    - edge_potentials: Python dictionary where `edge_potentials[(i, j)]` is
        a dictionaries-within-a-dictionary representation for a 2D potential
        table so that `edge_potentials[(i, j)][x_i][x_j]` corresponds to
        what, in the course notes, we call $\psi_{i,j}(x_i, x_j)$

        *IMPORTANT*: For the purposes of this project, please be sure to
        specify both `edge_potentials[(i, j)]` *and* `edge_potentials[(j, i)]`,
        where `edge_potentials[(i, j)][x_i][x_j]` should equal
        `edge_potentials[(j, i)][x_j][x_i]`

    Output
    ------
    - marginals: Python dictionary where `marginals[i]` gives the marginal
        distribution for node `i` represented as a dictionary; you do *not*
        need to store entries that are 0
    """
    marginals = {}
    messages = {}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    # edges is tree, find the leaves
    root = nodes.pop()
    nodes.add(root)

    def bfs(graph, nodes, root):
        queue_to_process = [root]
        marked = {node: False for node in nodes}
        # key is distance to root, val is list of node with the same distance
        # to root
        nodes_to_root_by_distance = {0: [root]}
        distance_to_root_by_node = {root: 0}
        # key is child node, val is parent node
        parent_of = {}

        while len(queue_to_process) > 0:
            node = queue_to_process.pop(0)
            marked[node] = True
            for neighbor in graph[node]:
                if not marked[neighbor]:
                    parent_of[neighbor] = node
                    distance = distance_to_root_by_node[node] + 1
                    distance_to_root_by_node[neighbor] = distance
                    if nodes_to_root_by_distance.get(distance) is None:
                        nodes_to_root_by_distance[distance] = [neighbor]
                    else:
                        nodes_to_root_by_distance[distance].append(neighbor)
                    queue_to_process.append(neighbor)

        return nodes_to_root_by_distance, distance_to_root_by_node, parent_of

    nodes_to_root_by_distance, distance_to_root_by_node, parent_of = bfs(
        edges, nodes, root)

    max_depth = max(nodes_to_root_by_distance.keys())
    visited = {node: False for node in nodes}

    # compute messages from leaves to root in the order of
    # distance to the root can avoid accessing messages that
    # haven't been computed yet, can you see why
    for depth in range(max_depth, 0, -1):
        nodes_to_porcess = nodes_to_root_by_distance[depth]
        for node in nodes_to_porcess:
            visited[node] = True
            # the only neighbor that will be processed
            # is the unique parent
            for neighbor in edges[node]:
                if not visited[neighbor]:
                    message_node_to_neighbor = {}
                    for xj in node_potentials[neighbor]:
                        message_node_to_neighbor[xj] = 0
                        for xi in node_potentials[node]:
                            message_xi = node_potentials[node][
                                xi] * edge_potentials[(node, neighbor)][xi][xj]
                            for other_neighbor in edges[node]:
                                message_other = messages.get(
                                    (other_neighbor, node))
                                if other_neighbor != neighbor and message_other is not None:
                                    message_xi *= message_other[xi]
                            message_node_to_neighbor[xj] += message_xi
                    messages[(node, neighbor)] = message_node_to_neighbor

    visited = {node: False for node in nodes}
    visited[root] = True
    for depth in range(1, max_depth + 1):
        nodes_to_process = nodes_to_root_by_distance[depth]
        for node in nodes_to_process:
            visited[node] = True
            for neighbor in edges[node]:
                if visited[neighbor]:
                    message_neighbor_to_node = {}
                    for xi in node_potentials[node]:
                        message_neighbor_to_node[xi] = 0
                        for xj in node_potentials[neighbor]:
                            message_xj = node_potentials[neighbor][
                                xj] * edge_potentials[(neighbor, node)][xj][xi]
                            for neighbors_neighbor in edges[neighbor]:
                                message_other = messages.get(
                                    (neighbors_neighbor, neighbor))
                                if neighbors_neighbor != node and message_other is not None:
                                    message_xj *= message_other[xj]
                            message_neighbor_to_node[xi] += message_xj
                    messages[(neighbor, node)] = message_neighbor_to_node

    for node in nodes:
        marginals[node] = {}
        for xi in node_potentials[node]:
            marginals[node][xi] = node_potentials[node][xi]
            for neighbor in edges[node]:
                marginals[node][xi] *= messages[(neighbor, node)][xi]
        Z = sum(list(marginals[node].values()))
        for key in marginals[node]:
            marginals[node][key] *= 1 / Z
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return marginals


def compute_marginals_given_observations(nodes, edges, node_potentials,
                                         edge_potentials, observations):
    """
    For a given choice of nodes, edges, node potentials, and edge potentials,
    and also observed values for specific nodes, we can compute marginals
    given the observations. This can actually be done by just modifying the
    node potentials and then calling the Sum-Product algorithm.

    Inputs
    ------
    - nodes, edges, node_potentials, edge_potentials: see documentation for
        sum_product()
    - observations: a dictionary where each key is a node and the value for
        the key is what the observed value for that node is (for example,
        `{1: 0}` means that node 1 was observed to have value 0)

    Output
    ------
    marginals, given the observations (see documentation for the output of
    sum_product())
    """
    new_node_potentials = {}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    for node in node_potentials:
        # new_node_potentials[node] refers to the same
        # object as node_potentials[node]
        new_node_potentials[node] = node_potentials[node].copy()
        if node in observations:
            ob = observations[node]
            for val in new_node_potentials[node]:
                # changing new_node_potentials[node][val]
                # changes node_potentials[node][val] too
                if val == ob:
                    new_node_potentials[node][val] = 1
                else:
                    new_node_potentials[node][val] = 0
    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return sum_product(nodes,
                       edges,
                       new_node_potentials,
                       edge_potentials)


# global variables to be saved for the trained classifier
guess = None


def train(windows, window_labels):
    """
    Your training procedure goes here! It should train a classifier where you
    store whatever you want to store for the trained classifier as *global*
    variables. `train` will get called exactly once on the exact same training
    data you have access to. However, you will not get access to the mystery
    test data.

    Inputs
    ------
    - windows, window_labels: see the documentation for the output of
        `get_labeled_windowed_data`
    """
    def classifier(function, nodes, edges, node_potentials, edge_potentials):
        def new_classifier(observations):
            return function(nodes, edges, node_potentials, edge_potentials, observations)
        return new_classifier
    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    windows_signs = np.sign(windows)
    num_windows = len(windows_signs)
    window_row, window_col = windows_signs[0].shape
    num_vars = window_row * window_col
    flattened_windows_signs = windows_signs.reshape((num_windows, num_vars))
    nodes = set([node for node in range(num_vars)])
    tree = chow_liu(flattened_windows_signs)
    (node_potentials, edge_potentials) = learn_tree_parameters(
        flattened_windows_signs, tree)
    # The autograder wants you to explicitly state which variables are global
    # and are supposed to thus be saved after training for use with prediction.
    global guess

    guess = classifier(compute_marginals_given_observations,
                       nodes, convert_tree_as_set_to_adjacencies(tree), node_potentials, edge_potentials)

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------


def forecast(window):
    """
    Your forecasting method goes here! You may assume that `train` has already
    been called on training data and so any global variables you stored as a
    result of running `train` are available to you here for prediction
    purposes.

    Input
    -----
    - window: 2D array; each column is 7 days worth of percent changes in
        price for a specific market

    Output
    ------
    1D array; the i-th entry is a prediction for whether the percentage
    return will go up (+1), stay the same (0), or go down (-1) for the i-th
    market
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    num_vars = window.shape[0] * window.shape[1]
    print("window: ", window)
    flattened_window = window.reshape(num_vars, )
    print("flattened_window: ", flattened_window)
    conditional_marginals = guess(flattened_window)
    row, col = window.shape
    row -= 1
    predicted_labels = []
    print("the true value: ", window[7])
    for idx in range(row * col, row * col + window.shape[1]):
        print("the marginals: ", conditional_marginals[idx])
        if conditional_marginals[idx][0.0] <= 0.7:
            marginal = conditional_marginals[idx].copy()
            marginal.pop(0.0)
            predicted_labels.append(max(marginal, key=marginal.get))
        else:
            predicted_labels.append(
                max(conditional_marginals[idx], key=conditional_marginals[idx].get))

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------
    return np.array(predicted_labels)


def main():
    # get coconut oil challenge training data
    observations = []
    with open('coconut_challenge.csv', 'r') as f:
        for line in f.readlines():
            pieces = line.split(',')
            if len(pieces) == 5:
                observations.append([float(pieces[1]),
                                     float(pieces[2]),
                                     float(pieces[3]),
                                     float(pieces[4])])
    observations = np.array(observations)
    train_windows, train_window_labels = \
        get_labeled_windowed_data(observations, window_size=7)

    train(train_windows, train_window_labels)

    # figure out accuracy of the trained classifier on predicting labels for
    # the training data
    train_predictions = []
    for window, window_label in zip(train_windows, train_window_labels):
        train_predictions.append(forecast(window))
    train_predictions = np.array(train_predictions)

    train_prediction_accuracy_plus1 = \
        np.mean(train_predictions[train_window_labels == 1]
                == train_window_labels[train_window_labels == 1])
    train_prediction_accuracy_minus1 = \
        np.mean(train_predictions[train_window_labels == -1]
                == train_window_labels[train_window_labels == -1])
    train_prediction_accuracy_0 = \
        np.mean(train_predictions[train_window_labels == 0]
                == train_window_labels[train_window_labels == 0])
    print('Training accuracy for prediction +1:',
          train_prediction_accuracy_plus1)
    print('Training accuracy for prediction -1:',
          train_prediction_accuracy_minus1)
    print('Training accuracy for prediction 0:',
          train_prediction_accuracy_0)


if __name__ == '__main__':
    main()
