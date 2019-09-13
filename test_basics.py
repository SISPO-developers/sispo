"""Test suite."""

import unittest
from pathlib import Path

import sispo.utils as utils


class TestUtils(unittest.TestCase):

    def test_resolve_create_dir(self):
        dir_name = "dir_test"

        file_dir = Path(__file__).parent.resolve()
        test_dir = file_dir / dir_name

        dir = utils.resolve_create_dir(test_dir)
        self.assertTrue(test_dir == dir)
        self.assertTrue(test_dir.exists())

        test_dir_relative = test_dir / ".." / dir_name
        dir = utils.resolve_create_dir(test_dir_relative)
        self.assertTrue(test_dir == dir)
        self.assertTrue(test_dir.exists())

        Path.rmdir(test_dir)


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
