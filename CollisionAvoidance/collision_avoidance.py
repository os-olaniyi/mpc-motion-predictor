from dataclasses import dataclass
from typing import List, Tuple, Optional
import casadi as ca
import numpy as np
import logging

from mpc import (
    NonlinearMPC,
    RobotState,
    MPCParams
    )

@dataclass
class Obstacle:
    """Representation of a circular obstacle"""
    x: float
    y: float
    radius: float

class CollisionAvoidanceMPC(NonlinearMPC):
    """MPC with collision avoidance capabilities"""
    
    def __init__(self, params: MPCParams, obstacles: List[Obstacle]):
        super().__init__(params)
        self.obstacles = obstacles
        self.solver = self._setup_solver()                                                              # Rebuild solver with collision avoidance
        self.logger = logging.getLogger(__name__)
    
    def _compute_obstacle_distances(self, x: ca.SX, y: ca.SX) -> List[ca.SX]:
        """Compute distances to all obstacles"""
        distances = []
        for obstacle in self.obstacles:
            # Calculate Euclidean distance to obstacle center
            dist = ca.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2) - obstacle.radius
            distances.append(dist)
        return distances
    
    def _setup_solver(self):
        """Setup CasADi solver with collision avoidance"""
        N = self.params.N
        dt = self.params.dt
        
        # State variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.size1()
        
        # Control variables
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        n_controls = controls.size1()
        
        # Define optimization variables
        X = ca.SX.sym('X', n_states, N + 1)                                                                 # states
        U = ca.SX.sym('U', n_controls, N)                                                                   # controls
        P = ca.SX.sym('P', n_states + 2*N)                                                                  # parameters (initial state + reference trajectory)
        
        # Initialize objective and constraints
        obj = 0
        g = []  # constraints vector
        
        # Initial constraints
        g.append(X[:, 0] - P[:n_states])
        
        # Loop over prediction horizon
        for k in range(N):
            # State cost
            ref_k = P[n_states + 2*k:n_states + 2*(k+1)]
            state_error = X[:2, k] - ref_k
            obj += self.params.state_weight * ca.mtimes(state_error.T, state_error)
            
            # Control cost
            control_input = U[:, k]
            obj += self.params.control_weight * ca.mtimes(control_input.T, control_input)
            
            # Collision avoidance cost
            obstacle_distances = self._compute_obstacle_distances(X[0, k], X[1, k])
            for dist in obstacle_distances:
                # Soft constraint using barrier function
                safety_distance = self.params.safety_distance
                obj += self.params.obstacle_weight * ca.exp(-dist + safety_distance)
                
                # Hard constraint for minimum safety distance
                g.append(dist)  # Must be greater than 0
            
            # Dynamics constraint (RK4)
            state_next = self._dynamics_rk4(X[:, k], U[:, k], dt)
            g.append(X[:, k+1] - state_next)
            
            # Control constraints
            g.append(U[0, k])  # velocity bounds
            g.append(U[1, k])  # angular velocity bounds
        
        # Terminal cost
        final_error = X[:2, -1] - P[n_states + 2*(N-1):n_states + 2*N]
        obj += self.params.terminal_weight * ca.mtimes(final_error.T, final_error)
        
        # Create optimization variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        # Define problem
        nlp = {
            'x': opt_vars,
            'f': obj,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        # Solver options
        opts = {
            'ipopt.print_level': 1 if self.params.debug else 0,
            'ipopt.max_iter': 500,  # Increased for better convergence
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.hessian_approximation': 'limited-memory',
            'print_time': 0,
            'verbose': self.params.debug,
            'expand': True
        }
        
        return ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def _dynamics_rk4(self, state, control, dt):
        """RK4 integration of system dynamics"""
        k1 = self._system_dynamics(state, control)
        k2 = self._system_dynamics(state + dt/2 * k1, control)
        k3 = self._system_dynamics(state + dt/2 * k2, control)
        k4 = self._system_dynamics(state + dt * k3, control)
        return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve(self, current_state: RobotState, reference_trajectory: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve the optimal control problem with collision avoidance"""
        try:
            N = self.params.N
            n_states = 3
            n_controls = 2
            n_obstacles = len(self.obstacles)
            
            # Initialize solution
            x0 = np.zeros((n_states * (N + 1) + n_controls * N, 1))
            x0[:n_states] = current_state.to_array().reshape(-1, 1)
            
            # Initialize parameters
            p = np.zeros(n_states + N * 2)
            p[:n_states] = current_state.to_array()
            
            # Fill reference trajectory
            for k in range(min(N, len(reference_trajectory))):
                p[n_states + 2*k:n_states + 2*(k+1)] = reference_trajectory[k]
            
            # Pad with last reference if needed
            if len(reference_trajectory) < N:
                for k in range(len(reference_trajectory), N):
                    p[n_states + 2*k:n_states + 2*(k+1)] = reference_trajectory[-1]
            
            # Set bounds for constraints
            lbg = []
            ubg = []
            
            # Initial state constraints
            lbg.extend([0.0] * n_states)
            ubg.extend([0.0] * n_states)
            
            # Constraints for each prediction step
            for _ in range(N):
                # Obstacle constraints (minimum distance)
                lbg.extend([0.0] * n_obstacles)  # Distance must be positive
                ubg.extend([float('inf')] * n_obstacles)  # No upper bound
                
                # Dynamics constraints
                lbg.extend([0.0] * n_states)
                ubg.extend([0.0] * n_states)
                
                # Control constraints
                lbg.extend([self.params.v_min, self.params.omega_min])
                ubg.extend([self.params.v_max, self.params.omega_max])
            
            # Solve optimization problem
            sol = self.solver(
                x0 = x0,
                lbg = lbg,
                ubg = ubg,
                p = p
            )
            
            # Extract solution
            solution = sol['x'].full().flatten()
            u0 = solution[n_states * (N + 1):n_states * (N + 1) + n_controls]
            x_pred = solution[:n_states * (N + 1)].reshape((N + 1, n_states))
            
            return u0, x_pred
            
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            return None, None