import random
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

class Orbit:
    def __init__(self, position: (float, float), veleocity: (float, float), seed: int = 0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(veleocity, dtype=float)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
       
    def is_visible(self, time: float) -> bool:
        # For now, random visibility based on time and seed
        return bool(self.rng.choice([True, False]))
    

class CelestialBody:
    def __init__(self, name, mass, position, velocity):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.orbit = Orbit(self.position, self.velocity)

class Sun(CelestialBody):
    def __init__(self, name, mass, position, velocity, orbit=None):
        super().__init__(name, mass, position, velocity)
        self.orbit = orbit

    def is_visible(self, time: float) -> bool:
        if self.orbit is not None:
            return self.orbit.is_visible(time)
        return True

class SolarSystem:
    def __init__(self, suns, planet_position):
        self.suns = suns  # List of CelestialBody (3 suns)
        self.planet_position = np.array(planet_position, dtype=float)  # Fixed

    def _three_sun_equations(self, t, state):
        G = 6.67430e-11
        m1, m2, m3 = [sun.mass for sun in self.suns]
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        r3 = np.array([x3, y3])
        r12 = r1 - r2
        r13 = r1 - r3
        r23 = r2 - r3
        dist12 = np.linalg.norm(r12)
        dist13 = np.linalg.norm(r13)
        dist23 = np.linalg.norm(r23)
        a1 = -G * m2 * r12 / dist12**3 - G * m3 * r13 / dist13**3
        a2 = -G * m1 * (-r12) / dist12**3 - G * m3 * (r2 - r3) / dist23**3
        a3 = -G * m1 * (-r13) / dist13**3 - G * m2 * (-r23) / dist23**3
        return [vx1, vy1, vx2, vy2, vx3, vy3, a1[0], a1[1], a2[0], a2[1], a3[0], a3[1]]

    def integrate(self, t_span, t_eval):
        # Initial state vector from suns
        state0 = []
        for sun in self.suns:
            state0.extend(sun.position)
        for sun in self.suns:
            state0.extend(sun.velocity)
        sol = solve_ivp(self._three_sun_equations, t_span, state0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
        return sol

def main():
    import matplotlib.pyplot as plt
    m_sun = 1.989e30
    AU = 1.496e11
    # Initial positions and velocities for three suns
    sun1 = Sun("Sun1", m_sun, [0.0, 2*AU], [0.0, 0.0])
    sun2 = Sun("Sun2", m_sun, [AU, 0.0], [0.0, 29.78e3])
    sun3 = Sun("Sun3", m_sun, [5*AU + 3.84e8, 0.0], [0.0, 29.78e3 + 1.022e3])
    solar_system = SolarSystem([sun1, sun2, sun3], [0.0, 0.0])
    t_span = (0, 3.154e7)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solar_system.integrate(t_span, t_eval)
    x1_sol, y1_sol = sol.y[0], sol.y[1]
    x2_sol, y2_sol = sol.y[2], sol.y[3]
    x3_sol, y3_sol = sol.y[4], sol.y[5]
    plt.figure(figsize=(8,8))
    plt.plot(x1_sol, y1_sol, 'y', label='Sun1')
    plt.plot(x2_sol, y2_sol, 'b', label='Sun2')
    plt.plot(x3_sol, y3_sol, 'r', label='Sun3')
    plt.scatter(x1_sol[0], y1_sol[0], color='y', marker='o')
    plt.scatter(x2_sol[0], y2_sol[0], color='b', marker='o')
    plt.scatter(x3_sol[0], y3_sol[0], color='r', marker='o')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Three-Body Problem: Three Suns')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()