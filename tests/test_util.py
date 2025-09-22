from util import contains_non_null_content


def test_no_content_is_null():
    assert not contains_non_null_content(None)


def test_empty_list_is_null():
    assert not contains_non_null_content([])


def test_empty_dict_is_null():
    assert not contains_non_null_content({})


def test_empty_leaves_is_null():
    assert not contains_non_null_content([{}, []])


def test_primitive_is_non_null():
    assert contains_non_null_content("haha")


def test_empty_string_is_non_null():
    assert contains_non_null_content("")


def test_partially_null_is_non_null():
    assert contains_non_null_content(["haha", None])


def test_nested_is_non_null():
    assert contains_non_null_content({"key": {"key": "value"}})
