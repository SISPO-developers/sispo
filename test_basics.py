"""Test suite."""

import unittest

import sispo.utils as utils


class TestUtils(unittest.TestCase):

    def test_write_vec_string(self):
        vec = [1.234, 2.345, 3.456]

        self.assertEqual(utils.write_vec_string(vec, 0), "[1,2,3]")
        self.assertEqual(utils.write_vec_string(vec, 1), "[1.2,2.3,3.5]")
        self.assertEqual(utils.write_vec_string(vec, 2), "[1.23,2.35,3.46]")
        self.assertEqual(utils.write_vec_string(vec, 3), "[1.234,2.345,3.456]")

        self.assertEqual(utils.write_vec_string(
            vec, 4), "[1.2340,2.3450,3.4560]")


if __name__ == "__main__":
    unittest.main()
