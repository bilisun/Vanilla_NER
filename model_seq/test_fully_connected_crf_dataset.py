from fully_connected_crf_dataset import get_char_len


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


if __name__ == "__main__":
    test_get_char_len()
