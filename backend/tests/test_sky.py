import pytest
from app.sky import sky

def test_sky_example():
    seed = 42
    sky_fn = sky(seed)
    time = 10.5
    visible = sky_fn(time)
    assert isinstance(visible, tuple)
    assert len(visible) == 3
    assert all(isinstance(v, bool) for v in visible)
    print(f"At time {time}, sun visibility (A, B, C): {visible}")
