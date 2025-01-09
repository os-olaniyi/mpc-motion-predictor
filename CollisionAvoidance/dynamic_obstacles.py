from dataclasses import dataclass
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import casadi as ca
from mpc import (
    RobotState,
    MPCParams,
    NonlinearMPC
    )
from collision_avoidance import (
    Obstacle
    )

@dataclass
class DynamicObstacle:
    """Representation of a moving obstacle"""
    x: float                  # Current x position
    y: float                  # Current y position
    vx: float                # Velocity in x direction
    vy: float                # Velocity in y direction
    radius: float            # Obstacle radius
    id: int                  # Unique identifier
    
    def predict_position(self, dt: float, time_horizon: float) -> np.ndarray:
        """Predict future positions of the obstacle"""
        num_steps = int(time_horizon / dt)
        positions = np.zeros((num_steps, 2))
        
        for i in range(num_steps):
            t = i * dt
            positions[i, 0] = self.x + self.vx * t
            positions[i, 1] = self.y + self.vy * t
            
        return positions
    
    def to_static_obstacle(self) -> Obstacle:
        """Convert to static obstacle for current position"""
        return Obstacle(x=self.x, y=self.y, radius=self.radius)

class DynamicObstacleTracker:
    """Tracks and predicts dynamic obstacles"""
    def __init__(self, prediction_horizon: float = 2.0, dt: float = 0.1):
        self.obstacles: List[DynamicObstacle] = []
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.next_id = 0
        self.obstacle_history: Dict[int, List[Tuple[float, float]]] = {}
        
    def add_obstacle(self, x: float, y: float, vx: float, vy: float, radius: float) -> int:
        """Add a new dynamic obstacle"""
        obstacle = DynamicObstacle(x=x, y=y, vx=vx, vy=vy, radius=radius, id=self.next_id)
        self.obstacles.append(obstacle)
        self.obstacle_history[self.next_id] = [(x, y)]
        self.next_id += 1
        return obstacle.id
    
    def update_obstacle(self, id: int, x: float, y: float, vx: float = None, vy: float = None):
        """Update obstacle position and optionally velocity"""
        for obstacle in self.obstacles:
            if obstacle.id == id:
                obstacle.x = x
                obstacle.y = y
                if vx is not None:
                    obstacle.vx = vx
                if vy is not None:
                    obstacle.vy = vy
                self.obstacle_history[id].append((x, y))
                break
    
    def predict_obstacles(self, time_horizon: Optional[float] = None) -> Dict[int, np.ndarray]:
        """Predict future positions of all obstacles"""
        if time_horizon is None:
            time_horizon = self.prediction_horizon
            
        predictions = {}
        for obstacle in self.obstacles:
            predictions[obstacle.id] = obstacle.predict_position(self.dt, time_horizon)
        return predictions
    
    def get_current_obstacles(self) -> List[Obstacle]:
        """Get current obstacles as static obstacles"""
        return [obs.to_static_obstacle() for obs in self.obstacles]

class DynamicCollisionAvoidanceMPC(NonlinearMPC):
    """MPC with dynamic obstacle avoidance capabilities"""
    
    def __init__(self, params: MPCParams, obstacle_tracker: DynamicObstacleTracker):
        # Store the obstacle tracker first
        self.obstacle_tracker = obstacle_tracker
        self.logger = logging.getLogger(__name__)
        
        # Then call parent initialization
        super().__init__(params)
        
        # Finally rebuild solver with collision avoidance
        self.solver = self._setup_solver()
    
    def _compute_dynamic_obstacle_distances(self, x: ca.SX, y: ca.SX, k: int) -> List[ca.SX]:
        """Compute distances to predicted obstacle positions at step k"""
        distances = []
        dt = self.params.dt
        
        for obstacle in self.obstacle_tracker.obstacles:
            # Predict obstacle position at time k*dt
            pred_x = obstacle.x + obstacle.vx * (k * dt)
            pred_y = obstacle.y + obstacle.vy * (k * dt)
            
            # Calculate distance to predicted position
            dist = ca.sqrt((x - pred_x)**2 + (y - pred_y)**2) - obstacle.radius
            distances.append(dist)
        
        return distances
    
    def _setup_solver(self):
        N = self.params.N
        dt = self.params.dt
        n_states = 3
        n_controls = 2
        
        # State variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        
        # Optimization variables
        X = ca.SX.sym('X', n_states, N + 1)
        U = ca.SX.sym('U', n_controls, N)
        P = ca.SX.sym('P', n_states + 2*N)
        
        obj = 0
        g = []
        
        # Initial state
        g.append(X[:, 0] - P[:n_states])
        
        for k in range(N):
            # Costs
            state_error = X[:2, k] - P[n_states + 2*k:n_states + 2*(k+1)]
            obj += self.params.state_weight * ca.mtimes(state_error.T, state_error)
            obj += self.params.control_weight * ca.mtimes(U[:, k].T, U[:, k])
            
            # Dynamic constraints
            x_next = self._dynamics_rk4(X[:, k], U[:, k], dt)
            g.append(X[:, k+1] - x_next)
            
            # Control bounds as constraints
            g.append(U[:, k])
        
        # Terminal cost
        final_error = X[:2, -1] - P[n_states + 2*(N-1):n_states + 2*N]
        obj += self.params.terminal_weight * ca.mtimes(final_error.T, final_error)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)
        
        return ca.nlpsol('solver', 'ipopt', 
                        {'x': opt_vars, 'f': obj, 'g': g, 'p': P},
                        {'ipopt': {'print_level': 0}})
    
    def _dynamics_rk4(self, state, control, dt):
        """RK4 integration of system dynamics"""
        k1 = self._system_dynamics(state, control)
        k2 = self._system_dynamics(state + dt/2 * k1, control)
        k3 = self._system_dynamics(state + dt/2 * k2, control)
        k4 = self._system_dynamics(state + dt * k3, control)
        return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve(self, current_state: RobotState, reference_trajectory: np.ndarray):
        try:
            N = self.params.N
            n_states = 3
            n_controls = 2
            
            # Initialize solution
            x0 = np.zeros((n_states * (N + 1) + n_controls * N, 1))
            x0[:n_states] = current_state.to_array().reshape(-1, 1)
            
            # Parameters vector
            p = np.zeros(n_states + N * 2)
            p[:n_states] = current_state.to_array()
            
            # Reference trajectory
            for k in range(min(N, len(reference_trajectory))):
                p[n_states + 2*k:n_states + 2*(k+1)] = reference_trajectory[k]
            if len(reference_trajectory) < N:
                for k in range(len(reference_trajectory), N):
                    p[n_states + 2*k:n_states + 2*(k+1)] = reference_trajectory[-1]
            
            # Constraint bounds
            lbg = []
            ubg = []
            
            # Initial state
            lbg.extend([0.0] * n_states)
            ubg.extend([0.0] * n_states)
            
            # Other constraints for each timestep
            for _ in range(N):
                # Dynamic constraints
                lbg.extend([0.0] * n_states)
                ubg.extend([0.0] * n_states)
                
                # Control bounds
                lbg.extend([self.params.v_min, self.params.omega_min])
                ubg.extend([self.params.v_max, self.params.omega_max])
            
            sol = self.solver(x0=x0, lbg=lbg, ubg=ubg, p=p)
            
            if self.solver.stats()['success']:
                solution = sol['x'].full().flatten()
                u0 = solution[n_states * (N + 1):n_states * (N + 1) + n_controls]
                x_pred = solution[:n_states * (N + 1)].reshape((N + 1, n_states))
                return u0, x_pred
                
            self.logger.error("Solver failed")
            return None, None
            
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            return None, None


class DynamicObstacleSimulator:
    """Simulates dynamic obstacles with different movement patterns"""
    
    def __init__(self, tracker: DynamicObstacleTracker):
        self.tracker = tracker
        
    def add_circular_obstacle(self, center_x: float, center_y: float, 
                            radius: float, angular_velocity: float, 
                            circle_radius: float) -> int:
        """Add an obstacle moving in a circular pattern"""
        # Initial position on the circle
        x = center_x + circle_radius
        y = center_y
        
        # Initial velocities
        vx = 0
        vy = angular_velocity * circle_radius
        
        return self.tracker.add_obstacle(x, y, vx, vy, radius)
    
    def add_linear_obstacle(self, start_x: float, start_y: float,
                          velocity_x: float, velocity_y: float,
                          radius: float) -> int:
        """Add an obstacle moving in a linear pattern"""
        return self.tracker.add_obstacle(start_x, start_y, velocity_x, velocity_y, radius)
    
    def update_obstacles(self, dt: float):
        """Update positions of all obstacles"""
        for obstacle in self.tracker.obstacles:
            # Update position based on velocity
            new_x = obstacle.x + obstacle.vx * dt
            new_y = obstacle.y + obstacle.vy * dt
            
            # If it's a circular obstacle, update velocities
            if abs(obstacle.vx**2 + obstacle.vy**2 - 
                  (obstacle.vx**2 + obstacle.vy**2)) < 1e-6:  # Constant speed check
                speed = np.sqrt(obstacle.vx**2 + obstacle.vy**2)
                angle = np.arctan2(obstacle.vy, obstacle.vx) + dt * speed
                obstacle.vx = speed * np.cos(angle)
                obstacle.vy = speed * np.sin(angle)
            
            self.tracker.update_obstacle(obstacle.id, new_x, new_y, obstacle.vx, obstacle.vy)