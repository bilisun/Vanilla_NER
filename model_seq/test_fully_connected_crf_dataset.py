from fully_connected_crf_dataset import get_char_len, to_onehot

import numpy as np


def test_onehot():
    indices = [3, 2, 0, 1]
    size = 5

    expected = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])

    np.testing.assert_array_equal(to_onehot(indices, size), expected)
    print('Test to_onehot passed!')


if __name__ == "__main__":
    test_get_char_len()
    test_onehot()
