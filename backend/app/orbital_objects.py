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
    def __init__(self, suns, planet):
        self.suns = suns  # List of CelestialBody (3 suns)
        self.planet = planet  # CelestialBody (planet)

    def _equations(self, t, state):
        G = 6.67430e-11
        m1, m2, m3 = [sun.mass for sun in self.suns]
        mp = self.planet.mass
        # Unpack state: [x1, y1, x2, y2, x3, y3, xp, yp, vx1, vy1, vx2, vy2, vx3, vy3, vxp, vyp]
        x1, y1, x2, y2, x3, y3, xp, yp, vx1, vy1, vx2, vy2, vx3, vy3, vxp, vyp = state
        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        r3 = np.array([x3, y3])
        rp = np.array([xp, yp])
        # Sun-sun vectors
        r12 = r1 - r2
        r13 = r1 - r3
        r23 = r2 - r3
        # Sun-planet vectors
        r1p = r1 - rp
        r2p = r2 - rp
        r3p = r3 - rp
        # Distances
        dist12 = np.linalg.norm(r12)
        dist13 = np.linalg.norm(r13)
        dist23 = np.linalg.norm(r23)
        dist1p = np.linalg.norm(r1p)
        dist2p = np.linalg.norm(r2p)
        dist3p = np.linalg.norm(r3p)
        # Accelerations for suns (include planet)
        a1 = -G * m2 * r12 / dist12**3 - G * m3 * r13 / dist13**3 - G * mp * r1p / dist1p**3
        a2 = -G * m1 * (-r12) / dist12**3 - G * m3 * (r2 - r3) / dist23**3 - G * mp * r2p / dist2p**3
        a3 = -G * m1 * (-r13) / dist13**3 - G * m2 * (-r23) / dist23**3 - G * mp * r3p / dist3p**3
        # Acceleration for planet (from all suns)
        ap = -G * m1 * (-r1p) / dist1p**3 - G * m2 * (-r2p) / dist2p**3 - G * m3 * (-r3p) / dist3p**3
        return [vx1, vy1, vx2, vy2, vx3, vy3, vxp, vyp, a1[0], a1[1], a2[0], a2[1], a3[0], a3[1], ap[0], ap[1]]

    def integrate(self, t_span, t_eval):
        # Initial state vector from suns and planet
        state0 = []
        for sun in self.suns:
            state0.extend(sun.position)
        state0.extend(self.planet.position)
        for sun in self.suns:
            state0.extend(sun.velocity)
        state0.extend(self.planet.velocity)
        sol = solve_ivp(self._equations, t_span, state0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
        return sol

def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    m_sun = 1.989e30
    m_earth = 5.972e24
    AU = 1.496e11
    sun1 = Sun("Sun1", m_sun, [0.0, 2*AU], [0.0, 0.0])
    sun2 = Sun("Sun2", m_sun, [1.5*AU, -1.5*AU], [0.0, 15000.0])
    sun3 = Sun("Sun3", m_sun, [-1.5*AU, -1.5*AU], [0.0, -15000.0])
    planet = CelestialBody("Planet", m_earth, [0, 0.0], [0.0, 20000.0])

    solar_system = SolarSystem([sun1, sun2, sun3], planet)
    t_span = (0, 3.154e7)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solar_system.integrate(t_span, t_eval)
    x1_sol, y1_sol = sol.y[0], sol.y[1]
    x2_sol, y2_sol = sol.y[2], sol.y[3]
    x3_sol, y3_sol = sol.y[4], sol.y[5]
    xp_sol, yp_sol = sol.y[6], sol.y[7]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Three Suns + Planet (Gravitational)')
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlim(np.min([x1_sol, x2_sol, x3_sol, xp_sol]), np.max([x1_sol, x2_sol, x3_sol, xp_sol]))
    ax.set_ylim(np.min([y1_sol, y2_sol, y3_sol, yp_sol]), np.max([y1_sol, y2_sol, y3_sol, yp_sol]))

    sun1_line, = ax.plot([], [], 'y-', label='Sun1')
    sun2_line, = ax.plot([], [], 'b-', label='Sun2')
    sun3_line, = ax.plot([], [], 'r-', label='Sun3')
    planet_line, = ax.plot([], [], 'g-', label='Planet')
    sun1_dot, = ax.plot([], [], 'yo')
    sun2_dot, = ax.plot([], [], 'bo')
    sun3_dot, = ax.plot([], [], 'ro')
    planet_dot, = ax.plot([], [], 'go')
    ax.legend()

    def init():
        sun1_line.set_data([], [])
        sun2_line.set_data([], [])
        sun3_line.set_data([], [])
        planet_line.set_data([], [])
        sun1_dot.set_data([], [])
        sun2_dot.set_data([], [])
        sun3_dot.set_data([], [])
        planet_dot.set_data([], [])
        return sun1_line, sun2_line, sun3_line, planet_line, sun1_dot, sun2_dot, sun3_dot, planet_dot

    def update(frame):
        sun1_line.set_data(x1_sol[:frame], y1_sol[:frame])
        sun2_line.set_data(x2_sol[:frame], y2_sol[:frame])
        sun3_line.set_data(x3_sol[:frame], y3_sol[:frame])
        planet_line.set_data(xp_sol[:frame], yp_sol[:frame])
        sun1_dot.set_data([x1_sol[frame-1]], [y1_sol[frame-1]])
        sun2_dot.set_data([x2_sol[frame-1]], [y2_sol[frame-1]])
        sun3_dot.set_data([x3_sol[frame-1]], [y3_sol[frame-1]])
        planet_dot.set_data([xp_sol[frame-1]], [yp_sol[frame-1]])
        return sun1_line, sun2_line, sun3_line, planet_line, sun1_dot, sun2_dot, sun3_dot, planet_dot

    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()