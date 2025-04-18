# Contains the physics logic for orbits of the three suns
# that surround the planet. 

# The three suns are named A, B, and C and orbit the planet. 

from typing import Tuple
from .orbital_objects import Sun, Orbit

def sky(seed: int):
    """
    Returns a function that, given a time, returns a tuple of booleans indicating
    whether suns A, B, and C are visible at that time.
    """
    # Initialize orbits and suns with the seed
    orbit_a = Orbit('A', seed + 1)
    orbit_b = Orbit('B', seed + 2)
    orbit_c = Orbit('C', seed + 3)
    sun_a = Sun('A', orbit_a)
    sun_b = Sun('B', orbit_b)
    sun_c = Sun('C', orbit_c)

    def at_time(time: float) -> Tuple[bool, bool, bool]:
        return (
            sun_a.is_visible(time),
            sun_b.is_visible(time),
            sun_c.is_visible(time)
        )
    return at_time


