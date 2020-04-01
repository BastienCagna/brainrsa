import unittest

from drsa.utils import tri_num, root_tri_num


# Tests for tri_num()
class TriNumFunction(unittest.TestCase):
    def test_tri_num_with_positive(self):
        self.assertEqual(tri_num(4), 10)

    def test_tri_num_output_type(self):
        self.assertEqual(type(tri_num(4)), int)

    def test_tri_num_with_negative_value(self):
        with self.assertRaises(ValueError):
            tri_num(-1)

    def test_tri_num_with_float_value(self):
        with self.assertRaises(ValueError):
            tri_num(2.5)
    

# Tests for root_tri_num
class RootTriNumFunction(unittest.TestCase):
    def test_root_tri_num_with_positive(self):
        self.assertEqual(root_tri_num(10), 4)

    def test_root_tri_num_output_type(self):
        self.assertEqual(type(root_tri_num(10)), int)

    def test_root_tri_num_with_negative_value(self):
        with self.assertRaises(ValueError):
            root_tri_num(-1)

    def test_root_tri_num_with_float_value(self):
        with self.assertRaises(ValueError):
            root_tri_num(2.5)

    def test_root_tri_num_with_non_trig_int(self):
        with self.assertRaises(ValueError):
            root_tri_num(8)


if __name__ == '__main__':
    unittest.main()

