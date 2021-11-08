import numpy as np
import pytest
from brainrsa.rdm import check_rdm

# Tests for tri_num()


def test_vector2matrix():
    vect = np.arange(0, 10)
    rdm = check_rdm(vect)
    assert rdm.shape == (5, 5)
    vect_r = check_rdm(rdm, force="vector")
    vect_equal = np.array_equal(np.nan_to_num(vect_r), np.nan_to_num(vect))
    assert vect_equal == True


def test_wrong_element_list():
    with pytest.raises(ValueError):
        assert check_rdm(np.arange(9))


def test_diagonal_filling():
    rdm = check_rdm(np.arange(0, 10), fill=-1, include_diag=False)
    for i in range(5):
        assert rdm[i, i] == -1


def test_only_lower():
    rdm = check_rdm(np.arange(3), fill=-1, triangle="lower")
    assert np.sum(rdm[np.triu_indices(3, k=0)]) == -6
    assert rdm[2, 0] == 1


def test_only_upper():
    rdm = check_rdm(np.arange(3), fill=-1, triangle="upper")
    assert np.sum(rdm[np.tril_indices(3, k=0)]) == -6
    assert rdm[0, 2] == 1


def test_output_types():
    assert check_rdm(np.arange(10, dtype=int), fill=0).dtype == int
    assert check_rdm(np.arange(10, dtype=float)).dtype == float
    assert check_rdm(np.arange(10, dtype=int)).dtype == int

# TODO: add test for include_diag and norm
