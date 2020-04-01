import unittest
import numpy as np

import unittest

from drsa.rdm import check_rdm


# Tests for tri_num()
class CheckRDMFunction(unittest.TestCase):
    
    def test_vector2matrix(self):
        vect = np.arange(0, 10)
        rdm = check_rdm(vect)
        self.assertEqual(rdm.shape, (5, 5))
        vect_r = check_rdm(rdm, force="vector")
        vect_equal = np.array_equal(np.nan_to_num(vect_r), np.nan_to_num(vect))
        self.assertEqual(vect_equal, True)

    def test_wrong_element_list(self):
        with self.assertRaises(ValueError):
            check_rdm(np.arange(9))
        
    def test_diagonal_filling(self):
        rdm = check_rdm(np.arange(0, 10), fill=-1, include_diag=False)
        for i in range(5):
            self.assertEqual(rdm[i, i], -1)

    def test_only_lower(self):
        rdm = check_rdm(np.arange(3), fill=-1, sigtri="lower")
        self.assertEqual(np.sum(rdm[np.triu_indices(3, k=0)]), -6)
        self.assertEqual(rdm[2, 0], 1)

    def test_only_upper(self):
        rdm = check_rdm(np.arange(3), fill=-1, sigtri="upper")
        self.assertEqual(np.sum(rdm[np.tril_indices(3, k=0)]), -6)
        self.assertEqual(rdm[0, 2], 1)
        
    def test_output_types(self):
        self.assertEqual(check_rdm(np.arange(10, dtype=int), fill=0).dtype, int)
        self.assertEqual(check_rdm(np.arange(10, dtype=float)).dtype, float)
        self.assertEqual(check_rdm(np.arange(10, dtype=int)).dtype, float)

   # TODO: add test for include_diag and norm


if __name__ == '__main__':
    unittest.main()


