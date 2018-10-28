from fully_connected_crf_dataset import get_char_len, to_onehot

import numpy as np


def test_get_char_len():
    c_con = 'c'
    word_limit = 3

    batch = [
            ['a', 'b', c_con, 'd', 'e', 'f', c_con, 'g', c_con, 'a', 'c', c_con],
            ['a', 'b', 'd', c_con, 'd', 'e', 'f', c_con, 'g', 'd', c_con],
            ['a', 'b', 'd', 'e', 'f', 'g', c_con],
            ['a', c_con, 'b', c_con],
    ]

    assert get_char_len(batch, c_con, word_limit) == 11
    print('Test get_char_len passed!')


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
