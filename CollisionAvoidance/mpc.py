from pathlib import Path
import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging
import time

# Logging Configuration
logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RobotState:
    """Robot state representation"""
    x: float
    y: float
    theta: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RobotState':
        return cls(x = float(arr[0]), y = float(arr[1]), theta = float(arr[2]))
    

@dataclass
class MPCParams:
    """MPC parameters"""
    N: int = 20                                                                             # Prediction horizon
    dt: float = 0.1                                                                         # Time step
    Q: np.ndarray = field(default_factory = lambda: np.diag([10.0, 10.0, 1.0]))             # State cost
    R: np.ndarray = field(default_factory = lambda: np.diag([0.1, 0.1]))                    # Control cost
    terminal_weight: float = 50.0
    state_weight: float = 10.0
    control_weight: float = 0.1
    v_max: float = 2.0
    v_min: float = -2.0
    omega_max: float = np.pi/2
    omega_min: float = -np.pi/2
    debug: bool = True
    safety_distance: float = 1.0                                                            # Minimum distance to keep from obstacles
    obstacle_weight: float = 10.0                                                           # Weight for obstacle avoidance in cost function

class NonlinearMPC:
    """MPC implementation using CasADi"""
    def __init__(self, params: MPCParams):
        self.params = params
        self.solver = self._setup_solver()
        self.logger = logging.getLogger(__name__)
        
    def _setup_solver(self):
        """Setup CasADi solver."""
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
        
        # RK4 integration
        def system_rk4(state, control, dt):
            k1 = self._system_dynamics(state, control)
            k2 = self._system_dynamics(state + dt/2 * k1, control)
            k3 = self._system_dynamics(state + dt/2 * k2, control)
            k4 = self._system_dynamics(state + dt * k3, control)
            return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Define variables for optimization
        X = ca.SX.sym('X', n_states, N + 1)                                                 # states
        U = ca.SX.sym('U', n_controls, N)                                                   # controls
        P = ca.SX.sym('P', n_states + 2*N)                                                  # parameters (initial state + reference trajectory)
        
        # Initialize objective and constraints
        obj = 0
        g = []                                                                              # constraints vector
        
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
            
            # Dynamics constraint (RK4)
            state_next = system_rk4(X[:, k], U[:, k], dt)
            g.append(X[:, k+1] - state_next)
            
            # Control constraints
            g.append(U[0, k])                                                               # velocity bounds
            g.append(U[1, k])                                                               # angular velocity bounds
        
        # Terminal cost
        final_error = X[:2, -1] - P[n_states + 2*(N-1):n_states + 2*N]
        obj += self.params.terminal_weight * ca.mtimes(final_error.T, final_error)
        
        # Create optimization variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        # Define optimization problem
        nlp = {
            'x': opt_vars,
            'f': obj,
            'g': ca.vertcat(*g),
            'p': P
        }
        
        # Solver options
        opts = {
            'ipopt.print_level': 1 if self.params.debug else 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.hessian_approximation': 'limited-memory',
            'print_time': 0,
            'verbose': self.params.debug,
            'expand': True
        }
        
        return ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def _system_dynamics(self, state, control):
        """Continuous time system dynamics"""
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]
        
        return ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega
        )
    
    def _debug_info(self, message: str, data: any):
        """Print debug information if debug mode is enabled"""
        if self.params.debug:
            self.logger.debug(f"{message}: {data}")

    def solve(self, 
             current_state: RobotState,
             reference_trajectory: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve the optimal control problem"""
        try:
            N = self.params.N
            n_states = 3
            n_controls = 2
            
            self._debug_info("Reference trajectory shape", reference_trajectory.shape)
            
            # Initialize solution
            x0 = np.zeros((n_states * (N + 1) + n_controls * N, 1))
            x0[:n_states] = current_state.to_array().reshape(-1, 1)
            
            # Initialize parameters
            p = np.zeros(n_states + N * 2)
            p[:n_states] = current_state.to_array()
            
            self._debug_info("Current state", current_state.to_array())
            
            # Fill reference trajectory
            for k in range(min(N, len(reference_trajectory))):
                p[n_states + 2*k:n_states + 2*(k+1)] = reference_trajectory[k]
                self._debug_info(f"Reference at step {k}", reference_trajectory[k])
            
            # Pad with last reference if needed
            if len(reference_trajectory) < N:
                for k in range(len(reference_trajectory), N):
                    p[n_states + 2*k:n_states + 2*(k+1)] = reference_trajectory[-1]
            
            # Set bounds
            lbg = np.zeros(n_states + N * (n_states + 2))
            ubg = np.zeros(n_states + N * (n_states + 2))
            
            # Control bounds indices
            control_start = n_states + N * n_states
            for k in range(N):
                vel_idx = control_start + 2*k
                lbg[vel_idx:vel_idx+2] = [self.params.v_min, self.params.omega_min]
                ubg[vel_idx:vel_idx+2] = [self.params.v_max, self.params.omega_max]
            
            self._debug_info("Problem dimensions",
                          f"x0: {x0.shape}, p: {p.shape}, lbg: {lbg.shape}")
            
            # Solve the optimization problem
            t_start = time.time()
            sol = self.solver(
                x0 = x0,
                lbg = lbg,
                ubg = ubg,
                p = p
            )
            solve_time = time.time() - t_start
            
            self._debug_info("Solve time", f"{solve_time:.3f} seconds")
            self._debug_info("Solver status", self.solver.stats()['return_status'])
            
            # Check solution
            if self.solver.stats()['success']:
                # Extract solution
                solution = sol['x'].full().flatten()
                
                # Get control and predicted trajectory
                u0 = solution[n_states * (N + 1):n_states * (N + 1) + n_controls]
                x_pred = solution[:n_states * (N + 1)].reshape((N + 1, n_states))
                
                self._debug_info("Control input", u0)
                self._debug_info("Prediction horizon", x_pred.shape)
                
                return u0, x_pred
            else:
                self.logger.warning("Solver failed to find a solution")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
            return None, None

def generate_reference_trajectory(num_points: int = 200) -> np.ndarray:
    """Generate a reference trajectory for the robot to follow"""
    t = np.linspace(0, 20, num_points)
    x_ref = t
    y_ref = 2 * np.sin(0.2 * t)
    return np.column_stack((x_ref, y_ref))

class Visualizer:
    """Handles visualization of robot state and trajectory"""
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize = (10, 8))
        
    def update(self, 
              robot_path: List[RobotState],
              reference_trajectory: np.ndarray,
              predicted_trajectory: Optional[np.ndarray] = None):
        """Update visualization with current state"""
        self.ax.clear()
        
        # Convert robot path to numpy array
        robot_positions = np.array([[state.x, state.y] for state in robot_path])
        
        # Plot reference trajectory
        self.ax.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], 
                    'g--', label = 'Reference Path', linewidth = 2)
        
        # Plot robot path
        self.ax.plot(robot_positions[:, 0], robot_positions[:, 1], 
                    'b-', label = 'Robot Path', linewidth = 2)
        
        # Plot current robot position
        if len(robot_positions) > 0:
            current_pos = robot_positions[-1]
            robot_circle = Circle(current_pos, 0.2, color = 'blue', alpha = 0.7)
            self.ax.add_patch(robot_circle)
        
        # Plot predicted trajectory if available
        if predicted_trajectory is not None:
            self.ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1],
                        'r:', label = 'Predicted Path', linewidth = 1)
        
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Robot Navigation with MPC')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.axis('equal')
        plt.pause(0.1)

########################################################
class EnhancedVisualizer:
    """Enhanced visualization with real-time metrics and safety zones"""
    def __init__(self):
        self.fig = plt.figure(figsize = (15, 10))
        self.gs = plt.GridSpec(2, 2)
        
        # Trajectory plot
        self.ax_traj = self.fig.add_subplot(self.gs[0, :])
        
        # Performance metrics plots
        self.ax_error = self.fig.add_subplot(self.gs[1, 0])
        self.ax_control = self.fig.add_subplot(self.gs[1, 1])
        
        plt.ion()  # Enable interactive mode
    
    def update(self,
              robot_path: List[RobotState],
              reference_trajectory: np.ndarray,
              predicted_trajectory: Optional[np.ndarray] = None,
              performance_metrics: Optional[Dict] = None,
              obstacles: Optional[List[Any]] = None,
              safety_distance: float = 1.0):
        """Update visualization with current state and metrics"""
        # Clear trajectory plot
        self.ax_traj.clear()
        
        # Plot reference and actual trajectories
        robot_positions = np.array([[state.x, state.y] for state in robot_path])
        self.ax_traj.plot(reference_trajectory[:, 0], reference_trajectory[:, 1],
                         'g--', label = 'Reference', linewidth = 2)
        self.ax_traj.plot(robot_positions[:, 0], robot_positions[:, 1],
                         'b-', label = 'Actual', linewidth = 2)
        
        # Plot current position and orientation
        if len(robot_positions) > 0:
            current_pos = robot_positions[-1]
            current_state = robot_path[-1]
            
            # Robot circle
            robot = plt.Circle(current_pos, 0.2, color = 'blue', alpha = 0.7)
            self.ax_traj.add_patch(robot)
            
            # Orientation arrow
            arrow_length = 0.5
            dx = arrow_length * np.cos(current_state.theta)
            dy = arrow_length * np.sin(current_state.theta)
            self.ax_traj.arrow(current_pos[0], current_pos[1], dx, dy,
                             head_width = 0.1, head_length = 0.2, fc = 'blue', ec = 'blue')
        
        # Plot predicted trajectory
        if predicted_trajectory is not None:
            self.ax_traj.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1],
                            'r:', label = 'Predicted', linewidth = 1)
        
        # Plot obstacles and safety zones
        if obstacles:
            for obstacle in obstacles:
                # Plot physical obstacle
                obstacle_circle = plt.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    color = 'red',
                    alpha = 0.5,
                    label = 'Obstacle' if obstacle == obstacles[0] else ""
                )
                self.ax_traj.add_patch(obstacle_circle)
                
                # Plot safety zone
                safety_circle = plt.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius + safety_distance,
                    color = 'yellow',
                    alpha = 0.2,
                    linestyle = '--',
                    fill = False,
                    label = 'Safety Zone' if obstacle == obstacles[0] else ""
                )
                self.ax_traj.add_patch(safety_circle)
                
                # Add optional text for dynamic obstacles
                if hasattr(obstacle, 'vx') and hasattr(obstacle, 'vy'):
                    velocity = np.sqrt(obstacle.vx**2 + obstacle.vy**2)
                    self.ax_traj.text(
                        obstacle.x, obstacle.y + obstacle.radius + 0.2,
                        f'v={velocity:.1f}m/s',
                        horizontalalignment='center'
                    )
        
        # Update trajectory plot settings
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Y Position (m)')
        self.ax_traj.set_title('Robot Navigation with Safety Zones')
        self.ax_traj.legend()
        self.ax_traj.grid(True)
        self.ax_traj.axis('equal')
        
        # Update performance metrics if available
        if performance_metrics:
            # Update error plot
            self.ax_error.clear()
            errors = performance_metrics['tracking_errors']
            self.ax_error.plot(errors, 'b-', label = 'Tracking Error')
            self.ax_error.set_title('Tracking Error')
            self.ax_error.set_ylabel('Error (m)')
            self.ax_error.grid(True)
            
            # Update control inputs plot
            self.ax_control.clear()
            controls = np.array(performance_metrics['control_inputs'])
            if len(controls) > 0:
                self.ax_control.plot(controls[:, 0], 'b-', label = 'v')
                self.ax_control.plot(controls[:, 1], 'r-', label = 'ω')
                self.ax_control.set_title('Control Inputs')
                self.ax_control.set_ylabel('Velocity')
                self.ax_control.grid(True)
                self.ax_control.legend()
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def plot_safety_zones(self, robot_path: List[RobotState], obstacles: List[Any], 
                         safety_distance: float):
        """Plot safety analysis visualization"""
        plt.figure(figsize = (12, 8))
        
        # Plot robot trajectory
        robot_positions = np.array([[state.x, state.y] for state in robot_path])
        plt.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', 
                label = 'Robot Path', linewidth = 2)
        
        # Plot obstacles and their safety zones
        for obstacle in obstacles:
            # Physical obstacle
            obstacle_circle = plt.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                color = 'red',
                alpha = 0.5
            )
            plt.gca().add_patch(obstacle_circle)
            
            # Safety zone
            safety_circle = plt.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius + safety_distance,
                color = 'yellow',
                alpha = 0.2,
                linestyle = '--',
                fill = False
            )
            plt.gca().add_patch(safety_circle)
            
            # Plot velocity vectors for dynamic obstacles
            if hasattr(obstacle, 'vx') and hasattr(obstacle, 'vy'):
                plt.arrow(
                    obstacle.x, obstacle.y,
                    obstacle.vx, obstacle.vy,
                    head_width = 0.1,
                    head_length = 0.2,
                    fc = 'gray',
                    ec = 'gray'
                )
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Path with Safety Zones')
        plt.grid(True)
        plt.axis('equal')
        plt.legend(['Robot Path', 'Obstacle', 'Safety Zone'])
        

class ConvergenceAnalyzer:
    """Analyzes convergence behavior of the MPC controller"""
    def __init__(self):
        self.metrics = {}
    
    def analyze_convergence(self, performance_monitor) -> Dict[str, Any]:
        """Perform statistical analysis of convergence"""
        # Get data
        errors = np.array(performance_monitor.metrics['tracking_errors'])
        controls = np.array(performance_monitor.metrics['control_inputs'])
        states = np.array([state.to_array() 
                          for state in performance_monitor.metrics['actual_states']])
        
        # Compute statistics
        self.metrics = {
            'error_stats': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'max': np.max(errors),
                'min': np.min(errors),
                'convergence_rate': self._compute_convergence_rate(errors)
            },
            'control_stats': {
                'v_mean': np.mean(controls[:, 0]),
                'v_std': np.std(controls[:, 0]),
                'w_mean': np.mean(controls[:, 1]),
                'w_std': np.std(controls[:, 1]),
                'control_effort': np.sum(np.square(controls))
            },
            'state_stats': {
                'position_error_mean': np.mean(np.linalg.norm(states[:, :2], axis = 1)),
                'orientation_error_mean': np.mean(np.abs(states[:, 2])),
                'final_position_error': np.linalg.norm(states[-1, :2]),
                'settling_time': self._compute_settling_time(errors)
            }
        }
        
        return self.metrics
    
    def _compute_convergence_rate(self, errors: np.ndarray, 
                                window: int = 10) -> float:
        """Compute rate of convergence using exponential fit"""
        if len(errors) < window:
            return 0.0
        
        # Use log of moving average
        ma = np.convolve(errors, np.ones(window)/window, mode = 'valid')
        ma = np.log(ma + 1e-6)  # Add small constant for numerical stability
        
        # Fit line to log data
        times = np.arange(len(ma))
        coeffs = np.polyfit(times, ma, 1)
        return -coeffs[0]  # Return negative slope as convergence rate
    
    def _compute_settling_time(self, errors: np.ndarray, 
                             threshold: float = 0.05) -> int:
        """Compute time to reach within threshold of final value"""
        final_value = errors[-1]
        threshold_value = final_value * (1 + threshold)
        
        # Find first index where error is consistently below threshold
        window = 10  # Number of consecutive points needed
        for i in range(len(errors) - window):
            if all(errors[i:i+window] < threshold_value):
                return i
        
        return len(errors)  # If never settles
    
class ConvergenceTuner:
    """Helps tune convergence parameters"""
    def __init__(self, nominal_params: Dict[str, float]):
        self.nominal_params = nominal_params
        self.results = {}
    
    def suggest_parameters(self, 
                         performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Suggest parameter adjustments based on performance"""
        suggestions = self.nominal_params.copy()
        
        # Analyze convergence behavior
        if performance_metrics['error_stats']['convergence_rate'] < 0.1:
            # Slow convergence - increase weights
            suggestions['state_weight'] *= 1.5
            suggestions['terminal_weight'] *= 1.2
        elif performance_metrics['control_stats']['control_effort'] > 1000:
            # Excessive control - reduce weights
            suggestions['control_weight'] *= 1.5
        
        # Adjust tolerances based on achieved accuracy
        final_error = performance_metrics['state_stats']['final_position_error']
        suggestions['position_tolerance'] = max(final_error * 0.1, 1e-4)
        
        return suggestions
    
    def evaluate_parameters(self, 
                          params: Dict[str, float], 
                          metrics: Dict[str, float]) -> float:
        """Compute parameter set performance score"""
        score = 0.0
        
        # Convergence speed
        score += metrics['error_stats']['convergence_rate'] * 10
        
        # Final accuracy
        score -= metrics['state_stats']['final_position_error']
        
        # Control effort penalty
        score -= metrics['control_stats']['control_effort'] * 0.001
        
        return score
    
        

class PerformanceMonitor:
    """Monitors and analyzes MPC performance metrics"""
    def __init__(self):
        self.metrics = {
            'solve_times': [],
            'tracking_errors': [],
            'control_inputs': [],
            'predicted_trajectories': [],
            'actual_states': [],       
            'actual_trajectories': [],   
            'reference_trajectories': [],
            'optimization_status': [],
            'constraint_violations': []  
        }
        self.logger = logging.getLogger(__name__)

    def record_iteration(self,
                    solve_time: float,
                    tracking_error: float,
                    control_input: np.ndarray,
                    predicted_trajectory: np.ndarray,
                    actual_state: RobotState,
                    reference_point: np.ndarray,
                    optimization_status: str):
        """Record metrics for one iteration"""
        self.metrics['solve_times'].append(solve_time)
        self.metrics['tracking_errors'].append(tracking_error)
        self.metrics['control_inputs'].append(control_input)
        self.metrics['predicted_trajectories'].append(predicted_trajectory)
        self.metrics['actual_states'].append(actual_state)                                      # Store RobotState object
        self.metrics['actual_trajectories'].append(actual_state.to_array())                     # Store array
        self.metrics['reference_trajectories'].append(reference_point)
        self.metrics['optimization_status'].append(optimization_status)
        
        # Calculate constraint violations
        control_violation = max(
            0,
            abs(control_input[0]) - 2.0,                                                    # v_max violation
            abs(control_input[1]) - np.pi/2                                                 # omega_max violation
        )
        self.metrics['constraint_violations'].append(control_violation)
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute performance statistics"""
        stats = {
            'avg_solve_time': np.mean(self.metrics['solve_times']),
            'max_solve_time': np.max(self.metrics['solve_times']),
            'avg_tracking_error': np.mean(self.metrics['tracking_errors']),
            'max_tracking_error': np.max(self.metrics['tracking_errors']),
            'success_rate': np.mean([status == 'Solve_Succeeded' 
                                   for status in self.metrics['optimization_status']]),
            'constraint_violation_rate': np.mean([v > 0 
                                                for v in self.metrics['constraint_violations']])
        }
        return stats
    
    def plot_performance_metrics(self, save_path: Optional[Path] = None):
        """Generate comprehensive performance plots"""
        if not self.metrics['solve_times']:                                                     # Check if we have data
            self.logger.warning("No Performance data to plot")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize = (15, 12))
        gs = plt.GridSpec(4, 2) 
        
        # 1. Tracking Error
        ax1 = fig.add_subplot(gs[0, 0])
        errors = np.array(self.metrics['tracking_errors'])
        times = np.array(range(len(errors)))
        ax1.plot(times, errors, 'b-', label = 'Tracking Error')
        ax1.axhline(y = np.mean(errors), color = 'r', linestyle = '--', 
                    label = f'Mean: {np.mean(errors):.2f}m')
        ax1.set_title('Tracking Error Over Time')
        ax1.set_ylabel('Error (m)')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Solve Times
        ax2 = fig.add_subplot(gs[0, 1])
        solve_times = np.array(self.metrics['solve_times']) * 1000                      # Convert to ms
        ax2.plot(times, solve_times, 'g-', label = 'Solve Time')
        ax2.axhline(y = np.mean(solve_times), color = 'r', linestyle = '--', 
                    label = f'Mean: {np.mean(solve_times):.2f}ms')
        ax2.set_title('Solver Performance')
        ax2.set_ylabel('Time (ms)')
        ax2.grid(True)
        ax2.legend()
        
        # 3. Control Inputs
        ax3 = fig.add_subplot(gs[1, :])
        controls = np.array(self.metrics['control_inputs'])
        if len(controls) > 0:
            ax3.plot(times, controls[:, 0], 'b-', label = 'Linear Velocity (v)')
            ax3.plot(times, controls[:, 1], 'r-', label = 'Angular Velocity (ω)')
            ax3.set_title('Control Inputs')
            ax3.set_ylabel('Velocity')
            ax3.grid(True)
            ax3.legend()
        
        # 4. State Evolution
        ax4 = fig.add_subplot(gs[2, :])
        if self.metrics['actual_states']:
            states = np.array([state.to_array() for state in self.metrics['actual_states']])
            ax4.plot(times, states[:, 0], 'r-', label = 'x position')
            ax4.plot(times, states[:, 1], 'g-', label = 'y position')
            ax4.plot(times, states[:, 2], 'b-', label = 'θ orientation')
            ax4.set_title('State Evolution')
            ax4.set_ylabel('State Values')
            ax4.grid(True)
            ax4.legend()
        
        # 5. Prediction Accuracy
        ax5 = fig.add_subplot(gs[3, :])
        if (len(self.metrics['predicted_trajectories']) > 0 and 
            len(self.metrics['actual_states']) > 0):
            predicted = np.array(self.metrics['predicted_trajectories'])
            actual_states = np.array([state.to_array() for state in self.metrics['actual_states']])
            pred_errors = np.linalg.norm(
                predicted[:, 0, :2] - actual_states[:, :2], axis = 1
            )
            ax5.plot(times, pred_errors, 'b-', label = 'Prediction Error')
            ax5.set_title('Prediction Accuracy')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Error (m)')
            ax5.grid(True)
            ax5.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_convergence_metrics(self, save_path: Optional[Path] = None):
        """Generate convergence analysis plots"""
        fig = plt.figure(figsize = (15, 12))
        gs = plt.GridSpec(3, 2)
        
        # 1. State Convergence
        ax1 = fig.add_subplot(gs[0, :])
        state_history = np.array([state.to_array() for state in self.metrics['actual_states']])
        times = range(len(state_history))
        
        ax1.plot(times, state_history[:, 0], 'r-', label = 'x position')
        ax1.plot(times, state_history[:, 1], 'g-', label = 'y position')
        ax1.plot(times, state_history[:, 2], 'b-', label = 'θ orientation')
        
        # Add moving average
        window = 10
        for i in range(3):
            ma = np.convolve(state_history[:, i], np.ones(window)/window, mode = 'valid')
            ax1.plot(times[window-1:], ma, '--', alpha = 0.5)
        
        ax1.set_title('State Convergence')
        ax1.grid(True)
        ax1.legend()

        # 2. Control Input Convergence
        ax2 = fig.add_subplot(gs[1, :])
        controls = np.array(self.metrics['control_inputs'])
        ax2.plot(times, controls[:, 0], 'b-', label = 'v')
        ax2.plot(times, controls[:, 1], 'r-', label = 'ω')
        
        # Add control limits
        ax2.axhline(y = 2.0, color = 'k', linestyle = '--', alpha = 0.3)
        ax2.axhline(y = -2.0, color = 'k', linestyle = '--', alpha = 0.3)
        ax2.axhline(y = np.pi/2, color = 'k', linestyle = '--', alpha = 0.3)
        ax2.axhline(y = -np.pi/2, color = 'k', linestyle = '--', alpha = 0.3)
        
        ax2.set_title('Control Input Convergence')
        ax2.grid(True)
        ax2.legend()

        # 3. Error Analysis
        ax3 = fig.add_subplot(gs[2, 0])
        errors = np.array(self.metrics['tracking_errors'])
        ax3.plot(times, errors, 'b-', label = 'Tracking Error')
        ax3.plot(times[window-1:], 
                np.convolve(errors, np.ones(window)/window, mode = 'valid'),
                'r--', label = f'Moving Avg (n = {window})')
        ax3.set_title('Tracking Error Convergence')
        ax3.set_yscale('log')
        ax3.grid(True)
        ax3.legend()

        # 4. Statistical Distribution
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(errors, bins = 30, density = True, alpha = 0.7)
        ax4.axvline(np.mean(errors), color = 'r', linestyle = '--', 
                    label = f'Mean: {np.mean(errors):.4f}')
        ax4.axvline(np.median(errors), color = 'g', linestyle = '--', 
                    label = f'Median: {np.median(errors):.4f}')
        ax4.set_title('Error Distribution')
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_report(self, save_path: Optional[Path] = None):
        """Generate a detailed performance report"""
        stats = self.get_statistics()
        
        # Calculate convergence metrics from existing data
        convergence_metrics = {
            'position_variation': np.std([state[:2] for state in self.metrics['actual_trajectories']], axis = 0).mean(),
            'orientation_variation': np.std([state[2] for state in self.metrics['actual_trajectories']]),
            'velocity_variation': np.std([control[0] for control in self.metrics['control_inputs']]),
            'angular_velocity_variation': np.std([control[1] for control in self.metrics['control_inputs']]),
            'mean_tracking_error': np.mean(self.metrics['tracking_errors'])
        }
        
        report = [
            "MPC Performance Report",
            "======================",
            f"Total Iterations: {len(self.metrics['solve_times'])}",
            f"Success Rate: {stats['success_rate']*100:.2f}%",
            "",
            "Convergence Metrics",
            "------------------",
            f"Position Variation: {convergence_metrics['position_variation']:.6f} m",
            f"Orientation Variation: {convergence_metrics['orientation_variation']:.6f} rad",
            f"Velocity Variation: {convergence_metrics['velocity_variation']:.6f} m/s",
            f"Angular Velocity Variation: {convergence_metrics['angular_velocity_variation']:.6f} rad/s",
            f"Mean Tracking Error: {convergence_metrics['mean_tracking_error']:.6f} m",
            "",
            "Timing Statistics",
            "-----------------",
            f"Average Solve Time: {stats['avg_solve_time']*1000:.2f} ms",
            f"Maximum Solve Time: {stats['max_solve_time']*1000:.2f} ms",
            "",
            "Tracking Performance",
            "-------------------",
            f"Average Error: {stats['avg_tracking_error']:.4f} m",
            f"Maximum Error: {stats['max_tracking_error']:.4f} m",
            f"Constraint Violation Rate: {stats['constraint_violation_rate']*100:.2f}%",
        ]
        
        report_text = "\n".join(report)
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text

  
class ConvergenceMonitor:
    """Monitors convergence of MPC optimization"""
    def __init__(self, position_tolerance = 1e-3, orientation_tolerance = 1e-2, 
                 control_tolerance = 1e-3, window_size = 5):
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.control_tolerance = control_tolerance
        self.window_size = window_size
        
        # History buffers
        self.state_history = []
        self.control_history = []
        self.error_history = []
        
        # Convergence flags
        self.state_converged = False
        self.control_converged = False
        self.error_converged = False
        
        self.logger = logging.getLogger(__name__)
    
    def update(self, current_state: RobotState, control_input: np.ndarray, 
               tracking_error: float) -> Dict[str, bool]:
        """Update convergence status"""
        # Add new data to history
        self.state_history.append(current_state.to_array())
        self.control_history.append(control_input)
        self.error_history.append(tracking_error)
        
        # Keep only recent history
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
            self.control_history.pop(0)
            self.error_history.pop(0)
        
        # Check convergence only if we have enough history
        if len(self.state_history) < self.window_size:
            return self._get_convergence_status()
        
        # Check state convergence
        state_variations = np.std(self.state_history, axis=0)
        self.state_converged = (
            np.all(state_variations[:2] < self.position_tolerance) and                              # position
            state_variations[2] < self.orientation_tolerance                                        # orientation
        )
        
        # Check control convergence
        control_variations = np.std(self.control_history, axis=0)
        self.control_converged = np.all(control_variations < self.control_tolerance)
        
        # Check error convergence
        error_variation = np.std(self.error_history)
        self.error_converged = error_variation < self.position_tolerance
        
        # Log convergence status
        self._log_convergence_status()
        
        return self._get_convergence_status()
    
    def _get_convergence_status(self) -> Dict[str, bool]:
        """Get current convergence status"""
        return {
            'state_converged': self.state_converged,
            'control_converged': self.control_converged,
            'error_converged': self.error_converged,
            'fully_converged': (self.state_converged and 
                              self.control_converged and 
                              self.error_converged)
        }
    
    def _log_convergence_status(self):
        """Log convergence information"""
        if len(self.state_history) == self.window_size:
            state_vars = np.std(self.state_history, axis = 0)
            control_vars = np.std(self.control_history, axis = 0)
            error_var = np.std(self.error_history)
            
            self.logger.debug(
                f"Convergence Status:\n"
                f"Position variation: {state_vars[:2]}\n"
                f"Orientation variation: {state_vars[2]}\n"
                f"Control variation: {control_vars}\n"
                f"Error variation: {error_var}"
            )
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get detailed convergence metrics"""
        if len(self.state_history) < 2:
            return {}
            
        state_vars = np.std(self.state_history, axis = 0)
        control_vars = np.std(self.control_history, axis = 0)
        
        return {
            'position_variation': float(np.mean(state_vars[:2])),
            'orientation_variation': float(state_vars[2]),
            'velocity_variation': float(control_vars[0]),
            'angular_velocity_variation': float(control_vars[1]),
            'error_variation': float(np.std(self.error_history)),
            'mean_tracking_error': float(np.mean(self.error_history))
        }