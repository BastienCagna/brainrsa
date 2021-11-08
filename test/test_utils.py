import pytest
from brainrsa.utils.misc import tri_num, root_tri_num


# Tests for tri_num()
def test_tri_num_with_positive():
    assert tri_num(4) == 10


def test_tri_num_output_type():
    assert type(tri_num(4)) == int


def test_tri_num_with_negative_value():
    with pytest.raises(ValueError):
        tri_num(-1)


def test_tri_num_with_float_value():
    with pytest.raises(ValueError):
        tri_num(2.5)


# Tests for root_tri_num
def test_root_tri_num_with_positive():
    assert root_tri_num(10) == 4


def test_root_tri_num_output_type():
    assert type(root_tri_num(10)) == int


def test_root_tri_num_with_negative_value():
    with pytest.raises(ValueError):
        root_tri_num(-1)


def test_root_tri_num_with_float_value():
    with pytest.raises(ValueError):
        root_tri_num(2.5)


def test_root_tri_num_with_non_trig_int():
    with pytest.raises(ValueError):
        root_tri_num(8)
