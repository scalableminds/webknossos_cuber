import inspect
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import pytest
from scipy.spatial import cKDTree

from webknossos.geometry import Mag


@contextmanager
def tmp_cwd() -> Iterator[None]:
    prev_cwd = os.getcwd()
    with TemporaryDirectory() as new_cwd:
        os.chdir(new_cwd)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


def exec_module_as_main(module: ModuleType) -> None:
    source = inspect.getsource(module)
    new_source = source.replace('\nif __name__ == "__main__":\n', "\nif True:\n")
    exec(new_source, module.__dict__)  # pylint: disable=exec-used


def test_dataset_usage() -> None:
    import examples.dataset_usage as example

    exec_module_as_main(example)

    assert example.data_in_mag1.shape == (3, 522, 532, 62)
    assert example.data_in_mag1_subset.shape == (3, 512, 512, 32)
    assert example.data_in_mag2.shape == (3, 261, 266, 31)
    assert example.data_in_mag2_subset.shape == (3, 256, 256, 16)


def test_skeleton_synapse_candidates() -> None:
    import examples.skeleton_synapse_candidates as example

    exec_module_as_main(example)

    assert example.synapse_parent_group.get_total_node_count() == 57
    ids = [g.id for g in example.nml.flattened_graphs()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Graph IDs are not unique."


@pytest.mark.vcr()
def test_upload_data() -> None:
    with tmp_cwd():
        import examples.upload_image_data as example

        exec_module_as_main(example)

        assert example.layer.bounding_box.size[0] == example.img.shape[1]
        assert example.layer.bounding_box.size[1] == example.img.shape[0]
        assert example.layer.bounding_box.size[2] == 1
        assert example.url.startswith(
            "http://localhost:9000/datasets/sample_organization/cell_"
        )


class _DummyNearestNeighborClassifier:
    """Faster replacement for a sklearn classifier,
    also removing the need for sklearn as a dependency."""

    labels: np.ndarray
    tree: cKDTree

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def fit(self, X: np.ndarray, labels: np.ndarray) -> None:
        MAX_FITTED_EXAMPLES = 64
        if X.shape[0] > MAX_FITTED_EXAMPLES:
            selection = np.random.default_rng(seed=42).choice(
                X.shape[0], MAX_FITTED_EXAMPLES, replace=False
            )
            self.labels = labels[selection]
            assert set(self.labels) == set(labels), (
                "Subsampling the examples omitted some labels, please use more examples.\n"
                + f"Currently MAX_FITTED_EXAMPLES is set to {MAX_FITTED_EXAMPLES}"
            )
            self.tree = cKDTree(X[selection])
        else:
            self.labels = labels
            self.tree = cKDTree(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        nearest_neighbors = self.tree.query(X)[1]
        return self.labels[nearest_neighbors]


@pytest.mark.vcr()
def test_learned_segmenter() -> None:
    with tmp_cwd():
        from skimage.future import trainable_segmentation

        old_default_classifier = None
        if trainable_segmentation.has_sklearn:
            old_default_classifier = trainable_segmentation.RandomForestClassifier
        trainable_segmentation.RandomForestClassifier = _DummyNearestNeighborClassifier
        trainable_segmentation.has_sklearn = True
        import examples.learned_segmenter as example

        exec_module_as_main(example)

        segmentation_data = example.segmentation_layer.mags[Mag(1)].read()
        counts = dict(zip(*np.unique(segmentation_data, return_counts=True)))
        assert counts == {1: 209066, 2: 37803, 3: 164553, 4: 817378}
        assert example.url.startswith(
            "http://localhost:9000/datasets/sample_organization/skin_segmented_"
        )

        if old_default_classifier is None:
            del trainable_segmentation.RandomForestClassifier
            trainable_segmentation.has_sklearn = False
        else:
            trainable_segmentation.RandomForestClassifier = old_default_classifier


@pytest.mark.vcr()
def test_user_times() -> None:
    import examples.user_times as example

    exec_module_as_main(example)

    assert len(example.df) > 0
    assert sum(example.df.loc[:, (2021, 10)]) > 0
    assert "taylor.tester@mail.com" in example.df.index
