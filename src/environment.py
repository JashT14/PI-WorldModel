import numpy as np
from typing import Tuple, Dict, Any, Optional, Union

class PendulumPhysicsEnv:
    """
    Continuous-time Non-Linear Inverted Pendulum Environment with Exact Work-Energy Accounting.
    
    Physics parameters:
      - m: mass of the pendulum bob (kg)
      - l: length of the massless rod (m)
      - g: acceleration due to gravity (m/s^2)
      - b: rotational friction damping coefficient (N*m*s/rad)
      - max_torque: maximum motor torque (N*m)
      - dt: simulation time step (s)
      - integrator: 'rk4' (Runge-Kutta 4th order) or 'symplectic_euler'
    
    Coordinate System:
      - theta: angle in radians (0 = upright top, pi / -pi = hanging down bottom)
      - theta_dot: angular velocity (rad/s)
      
    Observation:
      - [cos(theta), sin(theta), theta_dot]
    """
    def __init__(
        self,
        m: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        b: float = 0.1,
        max_torque: float = 2.5,
        dt: float = 0.05,
        integrator: str = "rk4"
    ):
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.max_torque = max_torque
        self.dt = dt
        self.integrator = integrator.lower()
        
        # Moment of inertia for point mass at rod end: I = m * l^2
        self.I = self.m * (self.l ** 2)
        
        # Cumulative work tracking
        self.cumulative_actuator_work = 0.0
        self.cumulative_damping_loss = 0.0
        
        # State: [theta, theta_dot]
        self.state = np.zeros(2, dtype=np.float32)
        self.reset()

    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment. Default is near bottom (theta ~ pi) with small noise.
        """
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float32)
        else:
            # Random angle around hanging position pi with small velocity
            theta = np.pi + np.random.uniform(-0.2, 0.2)
            theta_dot = np.random.uniform(-0.1, 0.1)
            self.state = np.array([theta, theta_dot], dtype=np.float32)
            
        self.cumulative_actuator_work = 0.0
        self.cumulative_damping_loss = 0.0
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """
        Returns continuous representation: [cos(theta), sin(theta), theta_dot]
        """
        theta, theta_dot = float(self.state[0]), float(self.state[1])
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def _angular_acceleration(self, theta: float, theta_dot: float, u: float) -> float:
        """
        Computes angular acceleration from Newton's 2nd Law for rotational dynamics:
        tau_net = I * alpha
        alpha = (m * g * l * sin(theta) + u - b * theta_dot) / (m * l^2)
        """
        gravity_torque = self.m * self.g * self.l * np.sin(theta)
        damping_torque = -self.b * theta_dot
        net_torque = gravity_torque + u + damping_torque
        return net_torque / self.I

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one physics step using the configured integrator (RK4 or Symplectic Euler).
        """
        u = float(np.clip(action, -self.max_torque, self.max_torque))
        theta, theta_dot = float(self.state[0]), float(self.state[1])
        dt = self.dt
        
        initial_energy = self.get_total_energy()

        if self.integrator == "symplectic_euler":
            alpha = self._angular_acceleration(theta, theta_dot, u)
            theta_dot_next = theta_dot + alpha * dt
            theta_next = theta + theta_dot_next * dt
        else: # Default: RK4
            def f(th: float, th_dot: float) -> Tuple[float, float]:
                return th_dot, self._angular_acceleration(th, th_dot, u)
            
            k1_th, k1_v = f(theta, theta_dot)
            k2_th, k2_v = f(theta + 0.5 * dt * k1_th, theta_dot + 0.5 * dt * k1_v)
            k3_th, k3_v = f(theta + 0.5 * dt * k2_th, theta_dot + 0.5 * dt * k2_v)
            k4_th, k4_v = f(theta + dt * k3_th, theta_dot + dt * k3_v)
            
            theta_next = theta + (dt / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
            theta_dot_next = theta_dot + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        
        # Track mechanical work done during step
        avg_vel = 0.5 * (theta_dot + theta_dot_next)
        step_actuator_work = u * avg_vel * dt
        step_damping_loss = self.b * (avg_vel ** 2) * dt
        self.cumulative_actuator_work += step_actuator_work
        self.cumulative_damping_loss += step_damping_loss

        # Normalize angle to [-pi, pi]
        theta_wrapped = ((theta_next + np.pi) % (2.0 * np.pi)) - np.pi
        self.state = np.array([theta_wrapped, theta_dot_next], dtype=np.float32)
        
        # Quadratic cost for upright stabilization (theta=0, theta_dot=0)
        angle_cost = float(theta_wrapped ** 2)
        velocity_cost = 0.1 * float(theta_dot_next ** 2)
        control_cost = 0.001 * float(u ** 2)
        reward = -(angle_cost + velocity_cost + control_cost)
        
        done = False
        final_energy = self.get_total_energy()
        
        info = {
            "theta": theta_wrapped,
            "theta_dot": theta_dot_next,
            "applied_torque": u,
            "kinetic_energy": self.get_kinetic_energy(),
            "potential_energy": self.get_potential_energy(),
            "total_energy": final_energy,
            "delta_energy": final_energy - initial_energy,
            "step_work": step_actuator_work,
            "step_damping": step_damping_loss
        }
        return self.get_observation(), reward, done, info

    def get_kinetic_energy(self) -> float:
        """T = 0.5 * I * theta_dot^2"""
        theta_dot = float(self.state[1])
        return 0.5 * self.I * (theta_dot ** 2)

    def get_potential_energy(self) -> float:
        """
        Potential energy relative to bottom (theta = pi):
        V(theta) = m * g * l * (1 + cos(theta))
        At upright top (theta = 0): V = 2 * m * g * l
        At hanging bottom (theta = pi): V = 0
        """
        theta = float(self.state[0])
        return self.m * self.g * self.l * (1.0 + np.cos(theta))

    def get_total_energy(self) -> float:
        return self.get_kinetic_energy() + self.get_potential_energy()


class CartPolePhysicsEnv:
    """
    Continuous-time Non-Linear Cart-Pole System on Cart with exact equations of motion.
    
    Physics parameters:
      - mc: mass of cart (kg)
      - mp: mass of pole (kg)
      - l: half-length of pole (m)
      - g: gravity (m/s^2)
      - max_force: maximum motor force on cart (N)
      - dt: simulation step (s)
      
    State: [x, x_dot, theta, theta_dot]
    Observation: [x, x_dot, cos(theta), sin(theta), theta_dot]
    """
    def __init__(
        self,
        mc: float = 1.0,
        mp: float = 0.1,
        l: float = 0.5,
        g: float = 9.81,
        max_force: float = 10.0,
        dt: float = 0.05
    ):
        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        self.max_force = max_force
        self.dt = dt
        self.total_mass = mc + mp
        self.pole_mass_length = mp * l
        
        self.state = np.zeros(4, dtype=np.float32)
        self.reset()

    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float32)
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        force = float(np.clip(action, -self.max_force, self.max_force))
        x, x_dot, theta, theta_dot = [float(v) for v in self.state]
        
        costh = np.cos(theta)
        sinth = np.sin(theta)
        
        temp = (force + self.pole_mass_length * (theta_dot ** 2) * sinth) / self.total_mass
        theta_acc = (self.g * sinth - costh * temp) / (
            self.l * (4.0 / 3.0 - self.mp * (costh ** 2) / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costh / self.total_mass
        
        # Symplectic Euler Step
        x_dot_next = x_dot + self.dt * x_acc
        x_next = x + self.dt * x_dot_next
        theta_dot_next = theta_dot + self.dt * theta_acc
        theta_next = theta + self.dt * theta_dot_next
        
        self.state = np.array([x_next, x_dot_next, theta_next, theta_dot_next], dtype=np.float32)
        reward = 1.0 - (theta ** 2 + 0.1 * (theta_dot ** 2) + 0.05 * (x ** 2))
        done = bool(abs(x_next) > 2.4 or abs(theta_next) > (np.pi / 2.0))
        
        return self.get_observation(), reward, done, {"x": x_next, "theta": theta_next}
