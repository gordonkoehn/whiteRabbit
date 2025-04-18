import numpy as np
import pytest
from app.orbital_objects import CelestialBody, Sun, SolarSystem, Orbit

def test_solar_system_three_suns():
    # Define three suns with different positions and velocities
    sun1 = CelestialBody('SunA', 1.989e30, [0.0, 0.0], [0.0, 0.0])
    sun2 = CelestialBody('SunB', 1.989e30, [1.496e11, 0.0], [0.0, 30000.0])
    sun3 = CelestialBody('SunC', 1.989e30, [0.0, 1.496e11], [-30000.0, 0.0])
    planet_position = [2.0e11, 2.0e11]  # Fixed, not used in force
    system = SolarSystem([sun1, sun2, sun3], planet_position)
    t_span = (0, 1e5)  # Short time for test
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    sol = system.integrate(t_span, t_eval)
    # Check that output shape is correct
    assert sol.y.shape[0] == 12  # 3 bodies * (x, y, vx, vy)
    assert sol.y.shape[1] == len(t_eval)
    # Check that positions change over time
    assert not np.allclose(sol.y[0], sol.y[0, 0])
    assert not np.allclose(sol.y[2], sol.y[2, 0])
    assert not np.allclose(sol.y[4], sol.y[4, 0])


def test_orbit_creation_and_visibility():
    pos = [1.0, 2.0]
    vel = [3.0, 4.0]
    orbit = Orbit(pos, vel, seed=42)
    assert np.allclose(orbit.position, pos)
    assert np.allclose(orbit.velocity, vel)
    # is_visible returns bool (random, but should not error)
    visible = orbit.is_visible(0.0)
    assert isinstance(visible, bool)

def test_celestial_body_initialization():
    body = CelestialBody('Test', 1.0, [1,2], [3,4])
    assert body.name == 'Test'
    assert body.mass == 1.0
    assert np.allclose(body.position, [1,2])
    assert np.allclose(body.velocity, [3,4])
    assert isinstance(body.orbit, Orbit)

def test_sun_inherits_celestial_body_and_visibility():
    sun = Sun('SunX', 2.0, [0,0], [1,1])
    assert isinstance(sun, CelestialBody)
    # is_visible should return a bool
    assert isinstance(sun.is_visible(0.0), bool)

def test_solar_system_integration_and_planet_fixed():
    mass = 1.989e30
    suns = [
        CelestialBody('A', mass, [0.0, 0.0], [0.0, 1e4]),
        CelestialBody('B', mass, [1.5e11, 0.0], [0.0, -1e4]),
        CelestialBody('C', mass, [0.0, 1.5e11], [1e4, 0.0]),
    ]
    planet_position = [2.5e11, 2.5e11]
    system = SolarSystem(suns, planet_position)
    t_span = (0, 1e5)
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    sol = system.integrate(t_span, t_eval)
    assert sol.y.shape[0] == 12
    assert sol.y.shape[1] == len(t_eval)
    assert np.allclose(system.planet_position, [2.5e11, 2.5e11])
