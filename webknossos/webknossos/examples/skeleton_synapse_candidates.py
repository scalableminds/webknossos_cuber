from itertools import combinations
from typing import Generator, Tuple

import numpy as np

import webknossos.skeleton as skeleton


def pairs_within_distance(
    pos_a: np.ndarray, pos_b: np.ndarray, max_distance: float
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    from scipy.spatial import cKDTree

    pos_a_kdtree = cKDTree(pos_a)
    pos_b_kdtree = cKDTree(pos_b)
    indexes = pos_a_kdtree.query_ball_tree(pos_b_kdtree, max_distance)
    for i in range(len(indexes)):
        for j in indexes[i]:
            yield (pos_a[i], pos_b[j])


def find_synapse_candidates():
    """
    Load an NML file and consider all pairs of trees.
    For each tree pair, find the node pairs that have a distance
    lower than a given threshold.
    For these candidates (with meaningful input data, these could be synapse candidates),
    new graphs are created which contain a node at the center position between the input
    nodes.
    """
    nml = skeleton.open_nml("../testdata/nmls/nml_with_small_distance_nodes.nml")

    synapse_candidate_max_distance = 0.5  # in nm

    input_graphs = nml.flattened_graphs()
    synapse_parent_group = nml.add_group("all synapse candidates")

    for tree_a, tree_b in combinations(input_graphs, 2):
        pos_a = tree_a.get_node_positions() * nml.scale
        pos_b = tree_b.get_node_positions() * nml.scale

        synapse_graph = synapse_parent_group.add_graph(
            f"synapse candidates ({tree_a.name}-{tree_b.name})"
        )

        for partner_a, partner_b in pairs_within_distance(
            pos_a, pos_b, synapse_candidate_max_distance
        ):
            synapse_graph.add_node(
                position=(partner_a + partner_b) / nml.scale / 2,
                comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
            )

    return nml, synapse_parent_group
