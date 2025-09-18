import jq
import pytest


@pytest.mark.skip(reason="Just for reference")
def test_jq():
    data = "1\n2\n3"
    result = jq.compile(".").input_value(data).first()
    pass
