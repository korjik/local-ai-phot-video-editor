import pytest

from app.image_utils import load_image


def test_load_image_rejects_unknown_bytes() -> None:
    with pytest.raises(ValueError, match="Could not decode"):
        load_image(b"not an image", max_side=512)
