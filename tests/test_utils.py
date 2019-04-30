import numpy as np
from wkcuber.utils import get_chunks, get_regular_chunks, BufferedSliceWriter

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
    bbox = {'topleft': (0, 0, 0), 'size': (24, 24, 35)}
    origin = (0, 0, 0)
    with BufferedSliceWriter.open('testoutput/buffered_slice_writer', 'color', np.uint16, bbox, origin) as writer:
