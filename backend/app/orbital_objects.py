import random
from typing import Tuple

class Orbit:
    def __init__(self, name: str, seed: int):
        self.name = name
        self.seed = seed
        self.rng = random.Random(seed)

    def is_visible(self, time: float) -> bool:
        # For now, random visibility based on time and seed
        return self.rng.choice([True, False])

class Sun:
    def __init__(self, name: str, orbit: Orbit):
        self.name = name
        self.orbit = orbit

    def is_visible(self, time: float) -> bool:
        return self.orbit.is_visible(time)
