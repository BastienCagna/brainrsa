import unittest

from brainrsa.inference import nperms_indexes


# Tests for tri_num()
class PermFunction(unittest.TestCase):
    def test_tri_num_with_positive(self):
        perms = nperms_indexes(5, 10)
        print(perms.shape)

#    def test_tri_num_output_type(self):
#        self.assertEqual(type(tri_num(4)), int)

#    def test_tri_num_with_negative_value(self):
#        with self.assertRaises(ValueError):
#            tri_num(-1)

#    def test_tri_num_with_float_value(self):
#        with self.assertRaises(ValueError):
#            tri_num(2.5)
    


if __name__ == '__main__':
    unittest.main()

