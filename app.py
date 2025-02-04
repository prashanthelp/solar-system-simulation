from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

app = Flask(__name__)

@app.route('/')
def simulate_solar_system():
    # Constants
    G = 6.67430e-11
    years_to_seconds = 365.25 * 24 * 60 * 60

    # Masses (kg)
    masses = {
        "Sun": 1.989e30,
        "Mercury": 3.301e23,
        "Venus": 4.867e24,
        "Earth": 5.972e24,
        "Moon": 7.342e22,
        "Mars": 6.417e23,
        "Jupiter": 1.898e27,
        "Saturn": 5.683e26,
        "Uranus": 8.681e25,
        "Neptune": 1.024e26,
        "Pluto": 1.309e22,
    }

    positions = {
        "Sun": np.array([0, 0, 0]),
        "Mercury": np.array([-5.790e10, 0, 0]),
        "Venus": np.array([-1.082e11, 0, 0]),
        "Earth": np.array([-1.496e11, 0, 0]),
        "Moon": np.array([-1.496e11 + 3.844e8, 0, 0]),
        "Mars": np.array([-2.279e11, 0, 0]),
        "Jupiter": np.array([-7.785e11, 0, 0]),
        "Saturn": np.array([-1.429e12, 0, 0]),
        "Uranus": np.array([-2.871e12, 0, 0]),
        "Neptune": np.array([-4.498e12, 0, 0]),
        "Pluto": np.array([-5.906e12, 0, 0]),
    }

    velocities = {
        "Sun": np.array([0, 0, 0]),
        "Mercury": np.array([0, 47.36e3, 0]),
        "Venus": np.array([0, 35.02e3, 0]),
        "Earth": np.array([0, 29.78e3, 0]),
        "Moon": np.array([0, 29.78e3 + 1.022e3, 0]),
        "Mars": np.array([0, 24.07e3, 0]),
        "Jupiter": np.array([0, 13.07e3, 0]),
        "Saturn": np.array([0, 9.69e3, 0]),
        "Uranus": np.array([0, 6.81e3, 0]),
        "Neptune": np.array([0, 5.43e3, 0]),
        "Pluto": np.array([0, 4.74e3, 0]),
    }

    bodies = list(masses.keys())
    n_bodies = len(bodies)
    y0 = np.concatenate([positions[body] for body in bodies] + [velocities[body] for body in bodies])

    def n_body_equations(t, y):
        r = y[:3 * n_bodies].reshape(n_bodies, 3)
        v = y[3 * n_bodies:].reshape(n_bodies, 3)
        a = np.zeros_like(r)
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_ij = r[j] - r[i]
                    r_mag = np.linalg.norm(r_ij)
                    a[i] += G * masses[bodies[j]] * r_ij / r_mag**3
        return np.concatenate([v.flatten(), a.flatten()])

    t_span = (0, 1 * years_to_seconds)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(n_body_equations, t_span, y0, method="RK45", t_eval=t_eval)
    positions_sim = sol.y[:3 * n_bodies].reshape(n_bodies, 3, -1)

    # Plot the orbits
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for i, body in enumerate(bodies):
        ax.plot(positions_sim[i, 0], positions_sim[i, 1], positions_sim[i, 2], label=body)
    ax.scatter([0], [0], [0], color="yellow", s=100, label="Sun")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.title("Solar System Simulation")

    # Save the plot
    image_path = os.path.join("static", "solar_system.png")
    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig(image_path)
    plt.close()

    return render_template('index.html', image_path=image_path)

if _name_ == '_main_':
    app.run(debug=True)