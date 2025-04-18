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
       
    def is_visible(self, planet_position: np.ndarray) -> bool:
        """Determine if the celestial body is visible from the planet's northern hemisphere.
           Returns True if the body's y > planet's y (i.e., in the northern hemisphere).
        """
        return self.position[1] > planet_position[1]
    
    def distance(self, other_position: np.ndarray) -> float:
        """Calculate the distance to another celestial body."""
        return np.linalg.norm(self.position - other_position)
    

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
        # estimate the radius of the sun based on its mass
        self.radius = (3 * mass / (4 * np.pi * 1.408e3)) ** (1/3)  # in meters, using average density of the sun
        self.temperature = 5778  # in Kelvin, average surface temperature of the sun

    def is_visible(self, planet_position: np.ndarray) -> bool:
        if self.orbit is not None:
            return self.orbit.is_visible(planet_position)
        raise RuntimeError("Sun must have an orbit set to determine visibility.")
    
    def distance(self, other_position: np.ndarray) -> float:
        if self.orbit is not None:
            return self.orbit.distance(other_position)
        raise RuntimeError("Sun must have an orbit set to calculate distance.")
    
    def power(self, other_position: np.ndarray) -> float:
        """Calculate the power received from the sun at the given position (W/m^2)."""
        distance = self.distance(other_position)
        if distance == 0:
            return float('inf')
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        luminosity = 4 * np.pi * self.radius**2 * sigma * self.temperature**4
        return luminosity / (4 * np.pi * distance**2)

class Planet(CelestialBody):
    def __init__(self, name, mass, position, velocity, albedo=0.5):
        super().__init__(name, mass, position, velocity)
        self.albedo = albedo  # reflectivity, default Earth-like

    def black_body_temperature(self, suns):
        """
        Calculate the equilibrium temperature of the planet based on black body radiation.
        suns: list of Sun objects
        Returns: temperature in Kelvin
        """
        sigma = 5.670374419e-8  # W/m^2/K^4
        total_power = 0.0
        for sun in suns:
            # Use Sun's power function for power received at planet's position
            power = sun.power(self.position)
            total_power += power
        absorbed_power = total_power * (1 - self.albedo)
        if absorbed_power <= 0:
            return 0.0
        temperature = (absorbed_power / sigma) ** 0.25
        return temperature

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

### Example usage

def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    m_sun = 1.989e30
    m_earth = 15.972e29
    AU = 1.496e11
    # Assign an Orbit to each Sun at creation
    sun1 = Sun("Sun1", m_sun, [0.0, 1.5*AU], [0.0, 0.0], orbit=Orbit([0.0, 2*AU], [0.0, 0.0]))
    sun2 = Sun("Sun2", m_sun, [1.5*AU, -1.5*AU], [0.0, 15000.0], orbit=Orbit([1.5*AU, -1.5*AU], [0.0, 15000.0]))
    sun3 = Sun("Sun3", m_sun, [-1.5*AU, -1.5*AU], [0.0, -15000.0], orbit=Orbit([-1.5*AU, -1.5*AU], [0.0, -15000.0]))
    planet = Planet("Planet", m_earth, [0, 0.0], [20000, 20000.0])

    solar_system = SolarSystem([sun1, sun2, sun3], planet)
    t_span = (0, 6.154e7)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solar_system.integrate(t_span, t_eval)
    x1_sol, y1_sol = sol.y[0], sol.y[1]
    x2_sol, y2_sol = sol.y[2], sol.y[3]
    x3_sol, y3_sol = sol.y[4], sol.y[5]
    xp_sol, yp_sol = sol.y[6], sol.y[7]

    fig, ax = plt.subplots(figsize=(8,8))
    # Set dark background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='gray', alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Three Suns + Planet (Gravitational)')
    ax.axis('equal')
    ax.set_xlim(np.min([x1_sol, x2_sol, x3_sol, xp_sol]), np.max([x1_sol, x2_sol, x3_sol, xp_sol]))
    ax.set_ylim(np.min([y1_sol, y2_sol, y3_sol, yp_sol]), np.max([y1_sol, y2_sol, y3_sol, yp_sol]))

    sun1_line, = ax.plot([], [], 'y-', label='Sun1')
    sun2_line, = ax.plot([], [], 'b-', label='Sun2')
    sun3_line, = ax.plot([], [], 'r-', label='Sun3')
    planet_line, = ax.plot([], [], 'g-', label='Planet')
    # Use empty lines for sun/planet markers, will update color in update()
    sun1_dot, = ax.plot([], [], 'o', markersize=10, label='Sun1 (visible)', color='orange')
    sun2_dot, = ax.plot([], [], 'o', markersize=10, label='Sun2 (visible)', color='orange')
    sun3_dot, = ax.plot([], [], 'o', markersize=10, label='Sun3 (visible)', color='orange')
    planet_dot, = ax.plot([], [], 'go', markersize=8, label='Planet')
    # Add legend for visible/invisible suns
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Sun (visible)', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Sun (not visible)', markerfacecolor='#8B0000', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Planet', markerfacecolor='lime', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    temperature_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, ha='left', va='top')

    def init():
        sun1_line.set_data([], [])
        sun2_line.set_data([], [])
        sun3_line.set_data([], [])
        planet_line.set_data([], [])
        sun1_dot.set_data([], [])
        sun2_dot.set_data([], [])
        sun3_dot.set_data([], [])
        planet_dot.set_data([], [])
        temperature_text.set_text('')
        return sun1_line, sun2_line, sun3_line, planet_line, sun1_dot, sun2_dot, sun3_dot, planet_dot, temperature_text

    def update(frame):
        sun1_line.set_data(x1_sol[:frame], y1_sol[:frame])
        sun2_line.set_data(x2_sol[:frame], y2_sol[:frame])
        sun3_line.set_data(x3_sol[:frame], y3_sol[:frame])
        planet_line.set_data(xp_sol[:frame], yp_sol[:frame])
        # Set sun marker color based on visibility
        planet_pos = np.array([xp_sol[frame-1], yp_sol[frame-1]])
        # Set orbits for each sun for visibility check
        sun1.orbit.position = np.array([x1_sol[frame-1], y1_sol[frame-1]])
        sun2.orbit.position = np.array([x2_sol[frame-1], y2_sol[frame-1]])
        sun3.orbit.position = np.array([x3_sol[frame-1], y3_sol[frame-1]])
        sun1_dot.set_data([x1_sol[frame-1]], [y1_sol[frame-1]])
        sun2_dot.set_data([x2_sol[frame-1]], [y2_sol[frame-1]])
        sun3_dot.set_data([x3_sol[frame-1]], [y3_sol[frame-1]])
        planet_dot.set_data([xp_sol[frame-1]], [yp_sol[frame-1]])
        # Color: orange if visible, dark red if not
        sun1_dot.set_color('orange' if sun1.is_visible(planet_pos) else '#8B0000')
        sun2_dot.set_color('orange' if sun2.is_visible(planet_pos) else '#8B0000')
        sun3_dot.set_color('orange' if sun3.is_visible(planet_pos) else '#8B0000')
        planet_dot.set_color('lime')
        # Dynamically center on planet and keep all objects in frame
        objects_x = [x1_sol[frame-1], x2_sol[frame-1], x3_sol[frame-1], xp_sol[frame-1]]
        objects_y = [y1_sol[frame-1], y2_sol[frame-1], y3_sol[frame-1], yp_sol[frame-1]]
        planet_x = xp_sol[frame-1]
        planet_y = yp_sol[frame-1]
        margin = 0.1 * max(np.ptp(objects_x), np.ptp(objects_y), 1e7)  # 10% margin, min margin
        x_min = min(objects_x)
        x_max = max(objects_x)
        y_min = min(objects_y)
        y_max = max(objects_y)
        # Center on planet, but expand to fit all objects
        ax.set_xlim(min(planet_x - (x_max-x_min)/2 - margin, x_min - margin),
                    max(planet_x + (x_max-x_min)/2 + margin, x_max + margin))
        ax.set_ylim(min(planet_y - (y_max-y_min)/2 - margin, y_min - margin),
                    max(planet_y + (y_max-y_min)/2 + margin, y_max + margin))
        # Calculate and display planet temperature
        planet.position = np.array([xp_sol[frame-1], yp_sol[frame-1]])
        sun1.position = np.array([x1_sol[frame-1], y1_sol[frame-1]])
        sun2.position = np.array([x2_sol[frame-1], y2_sol[frame-1]])
        sun3.position = np.array([x3_sol[frame-1], y3_sol[frame-1]])
        temperature = planet.black_body_temperature([sun1, sun2, sun3])
        temperature_text.set_text(f'Temperature: {temperature:.1f} K')
        return sun1_line, sun2_line, sun3_line, planet_line, sun1_dot, sun2_dot, sun3_dot, planet_dot, temperature_text

    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()