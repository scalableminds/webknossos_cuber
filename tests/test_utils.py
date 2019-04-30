import numpy as np
from wkcuber.utils import get_chunks, get_regular_chunks, BufferedSliceWriter
import wkw
from wkcuber.mag import Mag
import os

BLOCK_LEN = 32


def test_get_chunks():
    source = list(range(0, 48))
    target = list(get_chunks(source, 8))

    assert len(target) == 6
    assert target[0] == list(range(0, 8))


def test_get_regular_chunks():
    target = list(get_regular_chunks(4, 44, 8))

    assert len(target) == 6
    assert list(target[0]) == list(range(0, 8))
    assert list(target[-1]) == list(range(40, 48))


def test_get_regular_chunks_max_inclusive():
    target = list(get_regular_chunks(4, 44, 1))

    assert len(target) == 41
    assert list(target[0]) == list(range(4, 5))
    # The last chunk should include 44
    assert list(target[-1]) == list(range(44, 45))


def test_buffered_slice_writer():
    test_img = np.arange(24 * 24).reshape(24, 24).astype(np.uint16)
    dtype = test_img.dtype
    bbox = {'topleft': (0, 0, 0), 'size': (24, 24, 35)}
    origin = (0, 0, 0)
    dataset_dir = 'testoutput/buffered_slice_writer'
    layer_name = 'color'
    mag = Mag(1)
    dataset_path = os.path.join(dataset_dir, layer_name, mag.to_layer_name())

    with BufferedSliceWriter(dataset_dir, layer_name, dtype, bbox, origin, mag=mag) as writer:
        for i in range(13):
            writer.write_slice(test_img)
        with wkw.Dataset.open(dataset_path, wkw.Header(dtype)) as data:
            try:
                data.read(origin, (24, 24, 1))
                raise AssertionError('Nothing should be written on the disk.')
            except wkw.wkw.WKWException:
                pass
        for i in range(19):
            writer.write_slice(test_img)
        with wkw.Dataset.open(dataset_path, wkw.Header(dtype)) as data:
            read_data = data.read(origin, (24, 24, 32))
            assert read_data.shape == (24, 24, 32)
        for i in range(3):
            writer.write_slice(test_img)
    with wkw.Dataset.open(dataset_path, wkw.Header(dtype)) as data:
        read_data = data.read(origin, (24, 24, 35))
        assert read_data.shape == (24, 24, 35)


