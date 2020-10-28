#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy
import os
import chardet

def equals(check_value, expect_value, error_msg):
    """
    judge if check_value == expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert check_value == expect_value, error_msg


def less_than(check_value, expect_value, error_msg):
    """
    judge if check_value < expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert check_value < expect_value, error_msg


def less_than_or_equals(check_value, expect_value, error_msg):
    """
    judge if check_value <= expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert check_value <= expect_value, error_msg


def greater_than(check_value, expect_value, error_msg):
    """
    judge if check_value > expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert check_value > expect_value, error_msg


def greater_than_or_equals(check_value, expect_value, error_msg):
    """
    judge if check_value >= expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert check_value >= expect_value, error_msg


def not_equals(check_value, expect_value, error_msg):
    """
    judge if check_value != expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert check_value != expect_value, error_msg


def string_equals(check_value, expect_value, error_msg):
    """
    judge if str(check_value) == str(expect_value).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert str(check_value) == str(expect_value), error_msg


def length_equals(check_value, expect_value, error_msg):
    """
    judge if len(check_value) == expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert isinstance(expect_value, int)
    assert len(check_value) == expect_value, error_msg


def length_greater_than(check_value, expect_value, error_msg):
    """
    judge if len(check_value) > expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert isinstance(expect_value, int)
    assert len(check_value) > expect_value, error_msg


def length_greater_than_or_equals(check_value, expect_value, error_msg):
    """
    judge if len(check_value) >= expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert isinstance(expect_value, int)
    assert len(check_value) >= expect_value, error_msg


def length_less_than(check_value, expect_value, error_msg):
    """
    judge if len(check_value) < expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert isinstance(expect_value, int)
    assert len(check_value) < expect_value, error_msg


def length_less_than_or_equals(check_value, expect_value, error_msg):
    """
    judge if len(check_value) <= expect_value.
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:

    """
    assert isinstance(expect_value, int)
    assert len(check_value) <= expect_value, error_msg


def contains(check_value, expect_value, error_msg):
    """
    judge if str(expect_value) in str(check_value).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    if os.path.exists(check_value):
        with open(check_value) as f:
            check_value = f.read()
    assert str(expect_value) in str(check_value), "{} not in {}, check_value is {}".format(
        expect_value, check_value, check_value)


def not_contains(check_value, expect_value, error_msg):
    """
    judge if str(expect_value) not in str(check_value).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    if os.path.exists(check_value):
        with open(check_value) as f:
            check_value = f.read()
    assert str(expect_value) not in str(check_value), "{} in {}, check_value is {}".format(
        expect_value, check_value, check_value)


def type_match(check_value, expect_value, error_msg):
    """
    judge if isinstance(check_value, expect_value).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert isinstance(check_value, expect_value), error_msg


def startswith(check_value, expect_value, error_msg):
    """
    judge if str(check_value).startswith(str(expect_value)).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert str(check_value).startswith(str(expect_value)), error_msg


def endswith(check_value, expect_value, error_msg):
    """
    judge if str(check_value).endswith(str(expect_value)).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    assert str(check_value).endswith(str(expect_value)), error_msg


def regex_match_len(check_value, pattern, expect_value, error_msg):
    """
    judge if len(res) == expect_value.
    Args:
        check_value:
        pattern:
        expect_value:
        error_msg:

    Returns:
    """
    if os.path.exists(check_value):
        with open(check_value) as f:
            check_value = f.read()
    if isinstance(check_value, (bytearray, bytes)):
        encode_type = chardet.detect(check_value)
        check_value = check_value.decode(encode_type['encoding'])
    res = re.findall(pattern, check_value)
    print("res is {}".format(res))
    assert len(res) == expect_value, "{} != {}, check_value is:{}".format(len(res),
                                                                          expect_value,
                                                                          check_value)

def regex_match_close(check_value, pattern, expect_value, error_msg):
    """
    judge if numpy.allclose(res_arr, expect_value).
    Args:
        check_value:
        pattern:
        expect_value:
        error_msg:

    Returns:
    """
    if isinstance(check_value, (bytearray, bytes)):
        encode_type = chardet.detect(check_value)
        check_value = check_value.decode(encode_type['encoding'])
    res = re.findall(pattern, check_value)
    res_arr = numpy.array(res, dtype=float)
    assert numpy.allclose(res_arr, expect_value), "{} != {}, check_value is:{}".format(res_arr,
                                                                                       expect_value,
                                                                                       check_value)

def regex_match_equal(check_value, pattern, expect_value, error_msg):
    """

    Args:
        check_value:
        pattern:
        expect_value:
        error_msg:

    Returns:
    """
    matchObj = extract_value(check_value, pattern)
    assert matchObj
    if matchObj:
        assert str(matchObj[0]) == str(expect_value), "{} != {}, error is {}".format(
            matchObj[0], expect_value, error_msg)
    else:
        raise Exception("not found!")


def extract_value(check_value, pattern):
    """

    Args:
        check_value:
        pattern:
    Returns:
    """
    if isinstance(check_value, (bytearray, bytes)):
        encode_type = chardet.detect(check_value)
        check_value = check_value.decode(encode_type['encoding'])
    matchObj = re.findall(pattern, check_value)
    return matchObj


def numpy_close(check_value, expect_value, error_msg):
    """
    judge if numpy.allclose(res_arr, expect_value).
    Args:
        check_value:
        expect_value:
        error_msg:

    Returns:
    """
    check_value = numpy.array(check_value, dtype=float)
    expect_value = numpy.array(expect_value, dtype=float)
    print("check_value={}, expect_value={}".format(check_value, expect_value))
    assert numpy.allclose(check_value, expect_value), error_msg

def path_not_exist(path, error_msg):
    """
    judge path_not_exist.
    Args:
        path:
        error_msg:

    Returns:
    """
    assert not os.path.exists(path), "{} should not exist!, error is {}".format(path, error_msg)


def path_exist(path, error_msg):
    """
    judge path_exist.
    Args:
        path:
        error_msg:

    Returns:
    """
    assert os.path.exists(path), "{} should exist!, error is {}".format(path, error_msg)
