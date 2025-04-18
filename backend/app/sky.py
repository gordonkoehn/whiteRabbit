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
    # Use numeric positions/velocities for Orbit
    orbit_a = Orbit([0.0, 1.0], [0.0, 0.0], seed + 1)
    orbit_b = Orbit([1.0, 0.0], [0.0, 0.0], seed + 2)
    orbit_c = Orbit([0.0, -1.0], [0.0, 0.0], seed + 3)
    sun_a = Sun('A', 1.0, [0.0, 1.0], [0.0, 0.0], orbit=orbit_a)
    sun_b = Sun('B', 1.0, [1.0, 0.0], [0.0, 0.0], orbit=orbit_b)
    sun_c = Sun('C', 1.0, [0.0, -1.0], [0.0, 0.0], orbit=orbit_c)

    def at_time(time: float):
        # For demonstration, planet at (0,0)
        planet_pos = [0.0, 0.0]
        return (
            sun_a.is_visible(planet_pos),
            sun_b.is_visible(planet_pos),
            sun_c.is_visible(planet_pos)
        )
    return at_time


# TODO: implement a function to give the temperature of the surface


# TODO: add a fuciont of civilication death.


