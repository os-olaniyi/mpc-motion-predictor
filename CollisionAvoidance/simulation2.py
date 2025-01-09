from mpc import (
    RobotState, 
    MPCParams,
    NonlinearMPC,
    EnhancedVisualizer,
    PerformanceMonitor,
    generate_reference_trajectory,
    ConvergenceMonitor,
    ConvergenceTuner,
    ConvergenceAnalyzer
)

from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

import numpy as np
import yaml
import json
import logging
import matplotlib.pyplot as plt
import time

# Import new dynamic obstacle components
from collision_avoidance import (
    Obstacle
    )
from collision_detection import (
    CollisionDetector,
    SafetyMonitor
    )
from dynamic_obstacles import (
    DynamicObstacleTracker,
    DynamicObstacleSimulator,
    DynamicCollisionAvoidanceMPC
)

class SimulationEnvironment:
    """Main simulation environment integrating MPC, visualization, and performance monitoring"""
    
    def __init__(self, config_path: str = "config/simulation_config.yml"):
        """Initialize simulation environment"""
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Setup basic logging first
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level = logging.INFO,  
            format = '%(asctime)s - %(levelname)s - %(message)s',
            handlers = [
                logging.FileHandler('simulation.log'),
                logging.StreamHandler()
            ]
        )
    
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Update logging level based on config
        self.update_logging()
        
        # Initialize components
        self.setup_components()
        
        # Initialize data recording
        self.simulation_data = {
            'timestamps': [],
            'robot_states': [],
            'control_inputs': [],
            'reference_points': [],
            'tracking_errors': [],
            'computation_times': [],
            'predicted_trajectories': [],
            'obstacle_positions': [],
            'safety_metrics': []
        }
        
        # Initialize visualization and monitoring
        self.visualizer = EnhancedVisualizer()
        self.performance_monitor = PerformanceMonitor()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.convergence_tuner = ConvergenceTuner(self.config['mpc'])
        
        # Add convergence monitor
        self.convergence_monitor = ConvergenceMonitor(
            position_tolerance=self.config['convergence']['position_tolerance'],
            orientation_tolerance=self.config['convergence']['orientation_tolerance'],
            control_tolerance=self.config['convergence']['control_tolerance'],
            window_size=self.config['convergence']['window_size']
        )

    def update_logging(self):
        """Update logging level based on configuration"""
        log_level = logging.DEBUG if self.config.get('debug', {}).get('enabled', False) else logging.INFO
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def setup_components(self) -> None:
        """Initialize control system components"""
        try:
            # Create MPC parameters
            self.mpc_params = MPCParams(
                N = self.config['mpc']['horizon'],
                dt = self.config['mpc']['time_step'],
                Q = np.diag(self.config['mpc']['state_weights']),
                R = np.diag(self.config['mpc']['control_weights']),
                terminal_weight = self.config['mpc']['terminal_weight'],
                state_weight = self.config['mpc']['state_weight'],
                control_weight = self.config['mpc']['control_weight'],
                v_max = self.config['control_limits']['v_max'],
                v_min = self.config['control_limits']['v_min'],
                omega_max = self.config['control_limits']['omega_max'],
                omega_min = self.config['control_limits']['omega_min'],
                debug = self.config['debug']['enabled'],
                safety_distance = self.config['obstacles']['safety_distance'],
                obstacle_weight = self.config['obstacles']['weight']
            )
            
            # Initialize dynamic obstacle tracking
            self.obstacle_tracker = DynamicObstacleTracker(
                prediction_horizon = self.config['mpc']['horizon'] * self.config['mpc']['time_step']
            )
            
            # Initialize obstacle simulator
            self.obstacle_simulator = DynamicObstacleSimulator(self.obstacle_tracker)
            
            # Add dynamic obstacles from config
            self._setup_dynamic_obstacles()
            
            # Initialize MPC controller with dynamic obstacle avoidance
            self.mpc = DynamicCollisionAvoidanceMPC(self.mpc_params, self.obstacle_tracker)
            
            # Initialize collision detection and safety monitoring
            self.collision_detector = CollisionDetector(
                obstacles=self.obstacle_tracker.get_current_obstacles(),
                safety_distance=self.config['obstacles']['safety_distance']
            )
            self.safety_monitor = SafetyMonitor(self.collision_detector)
            
            self.logger.info("MPC controller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup components: {str(e)}")
            raise
    
    def _setup_dynamic_obstacles(self):
        """Setup dynamic obstacles from configuration"""
        self.logger.debug("Setting up dynamic obstacles")
        for obstacle_config in self.config['obstacles'].get('dynamic', []):
            if obstacle_config['type'] == 'circular':
                self.obstacle_simulator.add_circular_obstacle(
                    center_x = obstacle_config['center_x'],
                    center_y = obstacle_config['center_y'],
                    radius = obstacle_config['radius'],
                    angular_velocity = obstacle_config['angular_velocity'],
                    circle_radius = obstacle_config['circle_radius']
                )
                self.logger.debug(f"Added circular obstacle at ({obstacle_config['center_x']}, {obstacle_config['center_y']})")
            elif obstacle_config['type'] == 'linear':
                self.obstacle_simulator.add_linear_obstacle(
                    start_x = obstacle_config['start_x'],
                    start_y = obstacle_config['start_y'],
                    velocity_x = obstacle_config['velocity_x'],
                    velocity_y = obstacle_config['velocity_y'],
                    radius = obstacle_config['radius']
                )
                self.logger.debug(f"Added linear obstacle at ({obstacle_config['start_x']}, {obstacle_config['start_y']})")
    
    def _dynamics(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Simulate robot dynamics"""
        x, y, theta = state
        v, omega = control
        
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + omega * dt
        
        return np.array([x_next, y_next, theta_next])
    
    def record_step(self, timestamp: float, state: RobotState, control: np.ndarray,
                  reference: np.ndarray, predicted_traj: np.ndarray,
                  computation_time: float) -> None:
        """Record data from simulation step"""
        self.simulation_data['timestamps'].append(timestamp)
        self.simulation_data['robot_states'].append(state.to_array())
        self.simulation_data['control_inputs'].append(control)
        self.simulation_data['reference_points'].append(reference)
        self.simulation_data['predicted_trajectories'].append(predicted_traj)
        self.simulation_data['computation_times'].append(computation_time)
        
        # Record obstacle positions
        self.simulation_data['obstacle_positions'].append(
            [(obs.x, obs.y) for obs in self.obstacle_tracker.get_current_obstacles()]
        )
        
        # Calculate and record tracking error
        tracking_error = np.linalg.norm(state.to_array()[:2] - reference[:2])
        self.simulation_data['tracking_errors'].append(tracking_error)
    
    def run_simulation(self) -> None:
        """Run main simulation loop"""
        try:
            # Initialize simulation state
            current_state = RobotState(
                x = self.config['initial_state']['x'],
                y = self.config['initial_state']['y'],
                theta = self.config['initial_state']['theta']
            )
            
            # Generate reference trajectory
            reference_trajectory = generate_reference_trajectory(
                num_points = self.config['simulation']['num_points']
            )
            
            # Simulation parameters
            dt = self.config['mpc']['time_step']
            simulation_time = 0.0
            robot_path = [current_state]
            
            self.logger.info("Starting simulation...")
            self.logger.debug(f"Initial state: x = {current_state.x}, y = {current_state.y}, theta = {current_state.theta}")
            self.logger.debug(f"Reference trajectory points: {len(reference_trajectory)}")
            
            # Main simulation loop
            min_iterations = self.config['simulation']['min_iterations']
            for step in tqdm(range(self.config['simulation']['max_steps'])):
                # Update dynamic obstacles
                self.obstacle_simulator.update_obstacles(dt)
                self.collision_detector.obstacles = self.obstacle_tracker.get_current_obstacles()
                
                # Time step start
                step_start_time = time.time()
                
                # Get reference trajectory segment
                current_idx = min(step, len(reference_trajectory) - self.mpc_params.N)
                reference_segment = reference_trajectory[current_idx:current_idx + self.mpc_params.N]
                
                # Solve MPC problem
                control, predicted_trajectory = self.mpc.solve(
                    current_state,
                    reference_segment
                )
                
                if control is None:
                    self.logger.error("Failed to compute control input")
                    break
                
                self.logger.debug(f"Computed control: v = {control[0]:.4f}, omega = {control[1]:.4f}")
                
                # Update state
                next_state_array = self._dynamics(
                    current_state.to_array(),
                    control,
                    dt
                )
                next_state = RobotState.from_array(next_state_array)
                
                self.logger.debug(f"Next state: x = {next_state.x:.4f}, y = {next_state.y:.4f}, theta = {next_state.theta:.4f}")
                
                # Calculate performance metrics
                computation_time = time.time() - step_start_time
                tracking_error = np.linalg.norm(
                    next_state.to_array()[:2] - reference_segment[0]
                )
                
                # Update safety monitoring
                safety_status = self.safety_monitor.update(
                    current_state = current_state,
                    predicted_trajectory = predicted_trajectory,
                    control_input = control
                )
                
                self.logger.debug(f"Tracking error: {tracking_error:.4f}")
                self.logger.debug(f"Computation time: {computation_time*1000:.2f}ms")
                
                # Monitor convergence
                convergence_status = self.convergence_monitor.update(
                    current_state = current_state,
                    control_input = control,
                    tracking_error = tracking_error
                )
                
                # Record step data
                self.record_step(
                    timestamp = simulation_time,
                    state = current_state,
                    control = control,
                    reference = reference_segment[0],
                    predicted_traj = predicted_trajectory,
                    computation_time = computation_time
                )
                
                # Update performance monitor
                self.performance_monitor.record_iteration(
                    solve_time = computation_time,
                    tracking_error = tracking_error,
                    control_input = control,
                    predicted_trajectory = predicted_trajectory,
                    actual_state = current_state,
                    reference_point = reference_segment[0],
                    optimization_status = self.mpc.solver.stats()['return_status']
                )
                
                # Update visualization
                if step % self.config['visualization']['update_interval'] == 0:
                    self.visualizer.update(
                        robot_path = robot_path,
                        reference_trajectory = reference_trajectory,
                        predicted_trajectory = predicted_trajectory,
                        performance_metrics = self.performance_monitor.metrics,
                        obstacles = self.obstacle_tracker.get_current_obstacles(),
                        safety_distance = self.config['obstacles']['safety_distance']
                    )
                    self.logger.debug(f"Updated visualization at step {step}")
                
                # Log convergence status periodically
                if step % 10 == 0:
                    metrics = self.convergence_monitor.get_convergence_metrics()
                    self.logger.info(f"Convergence metrics at step {step}:")
                    for key, value in metrics.items():
                        self.logger.info(f"  {key}: {value:.6f}")
                
                # Check completion
                if (step >= min_iterations and 
                    tracking_error < self.config['simulation']['completion_threshold']):
                    self.logger.info(f"Reached target with desired accuracy after {step} iterations")
                    self.logger.debug(f"Final tracking error: {tracking_error:.4f}")
                    break
                
                # Update state and time
                current_state = next_state
                robot_path.append(current_state)
                simulation_time += dt
                
                if current_idx >= len(reference_trajectory) - self.mpc_params.N - 1:
                    self.logger.info("Reached end of reference trajectory")
                    break
            
            # Generate analysis
            convergence_metrics = self.convergence_analyzer.analyze_convergence(
                self.performance_monitor
            )
            
            # Generate suggestions for next run
            suggested_params = self.convergence_tuner.suggest_parameters(
                convergence_metrics
            )
            
            # Save analysis results
            self.save_convergence_analysis(convergence_metrics, suggested_params)
            
            # Log final statistics
            self.logger.info(f"Simulation completed with {len(robot_path)} steps")
            self.logger.info(f"Final position: x = {current_state.x:.4f}, y = {current_state.y:.4f}, theta = {current_state.theta:.4f}")
            self.logger.info(f"Average tracking error: {np.mean(self.simulation_data['tracking_errors']):.4f}")
            
            self.save_results()
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            raise

    def save_convergence_analysis(self, metrics: Dict, suggested_params: Dict):
        """Save convergence analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = Path("results") / f"analysis_{timestamp}"
        analysis_dir.mkdir(parents = True, exist_ok = True)
        
        # Save metrics
        with open(analysis_dir / "convergence_metrics.json", 'w') as f:
            json.dump(metrics, f, indent = 2)
        
        # Save parameter suggestions
        with open(analysis_dir / "parameter_suggestions.json", 'w') as f:
            json.dump(suggested_params, f, indent = 2)
        
        # Generate convergence plots
        self.performance_monitor.plot_convergence_metrics(
            save_path=analysis_dir / "convergence_plots.png"
        )
    
    def save_results(self) -> None:
        """Save simulation results"""
        try:
            # Create result directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = Path("results") / f"simulation_{timestamp}"
            result_dir.mkdir(parents = True, exist_ok = True)
            
            # Save complete trajectory plot
            plt.figure(figsize = (15, 10))
            
            # Convert trajectories to numpy arrays for plotting
            robot_states = np.array(self.simulation_data['robot_states'])
            reference_points = np.array(self.simulation_data['reference_points'])
            
            # Plot reference trajectory
            plt.plot(reference_points[:, 0], reference_points[:, 1], 
                    'g--', label = 'Reference', linewidth = 2)
            
            # Plot actual trajectory
            plt.plot(robot_states[:, 0], robot_states[:, 1], 
                    'b-', label = 'Actual', linewidth = 2)
            
             # Plot obstacles and safety zones
            obstacles = self.obstacle_tracker.get_current_obstacles()
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
                    obstacle.radius + self.config['obstacles']['safety_distance'],
                    color = 'yellow',
                    alpha = 0.2,
                    linestyle = '--',
                    fill = False
                )
                plt.gca().add_patch(safety_circle)
            
            # Plot start and end points
            plt.scatter(robot_states[0, 0], robot_states[0, 1], 
                       c = 'g', marker = 'o', s = 100, label = 'Start')
            plt.scatter(robot_states[-1, 0], robot_states[-1, 1], 
                       c = 'r', marker = 'x', s = 100, label = 'End')
            
            # Plot final predicted trajectory if available
            if self.simulation_data['predicted_trajectories']:
                final_prediction = self.simulation_data['predicted_trajectories'][-1]
                if final_prediction is not None:
                    plt.plot(final_prediction[:, 0], final_prediction[:, 1], 
                            'r:', label = 'Final Prediction', linewidth = 1)
            
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title('Complete Robot Trajectory')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.savefig(result_dir / "complete_trajectory.png")
            plt.close()
            
            # Save trajectory data as NPZ
            np.savez(
                result_dir / "trajectory_data.npz",
                robot_states = np.array(self.simulation_data['robot_states']),
                obstacle_positions = np.array(self.simulation_data['obstacle_positions']),
                reference_points = np.array(self.simulation_data['reference_points']),
                tracking_errors = np.array(self.simulation_data['tracking_errors'])
            )
            
            # Save as JSON for human-readable format
            trajectory_data = {
                'timestamps': [float(t) for t in self.simulation_data['timestamps']],
                'robot_states': [state.tolist() for state in self.simulation_data['robot_states']],
                'reference_points': [ref.tolist() for ref in self.simulation_data['reference_points']],
                'obstacle_positions': self.simulation_data['obstacle_positions'],
                'tracking_errors': [float(error) for error in self.simulation_data['tracking_errors']],
                'computation_times': [float(time) for time in self.simulation_data['computation_times']]
            }
            
            with open(result_dir / "trajectory_data.json", 'w') as f:
                json.dump(trajectory_data, f, indent = 2)
            
            # Save performance metrics
            self.performance_monitor.plot_performance_metrics(
                save_path = result_dir / "performance_metrics.png"
            )
            
            # Generate and save performance report
            self.performance_monitor.generate_report(
                save_path = result_dir / "performance_report.txt"
            )
            
            # Save collision and safety metrics
            safety_metrics = {
                'safety_statistics': self.collision_detector.get_safety_statistics(),
                'safety_monitor_metrics': self.safety_monitor.get_safety_metrics()
            }
            
            with open(result_dir / "safety_metrics.json", 'w') as f:
                json.dump(safety_metrics, f, indent = 2)
            
            # Save simulation summary
            summary = {
                'total_steps': len(robot_states),
                'final_position': robot_states[-1].tolist(),
                'total_distance': float(np.sum(np.linalg.norm(np.diff(robot_states[:, :2], axis = 0), axis = 1))),
                'average_tracking_error': float(np.mean(self.simulation_data['tracking_errors'])),
                'max_tracking_error': float(np.max(self.simulation_data['tracking_errors'])),
                'average_computation_time': float(np.mean(self.simulation_data['computation_times'])),
                'total_simulation_time': float(self.simulation_data['timestamps'][-1])
            }
            
            with open(result_dir / "simulation_summary.json", 'w') as f:
                json.dump(summary, f, indent = 2)
            
            self.logger.info(f"Saved simulation results to {result_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def analyze_results(self) -> None:
        """Analyze simulation results"""
        robot_states = np.array(self.simulation_data['robot_states'])
        tracking_errors = np.array(self.simulation_data['tracking_errors'])
        
        # Print statistics
        print("\nSimulation Analysis:")
        print(f"Total steps: {len(robot_states)}")
        print(f"Final position: ({robot_states[-1, 0]:.2f}, {robot_states[-1, 1]:.2f})")
        print(f"Average tracking error: {np.mean(tracking_errors):.4f} m")
        print(f"Maximum tracking error: {np.max(tracking_errors):.4f} m")
        
        # Print safety statistics
        safety_stats = self.collision_detector.get_safety_statistics()
        print("\nSafety Analysis:")
        print(f"Minimum distance to obstacles: {safety_stats['minimum_distance_ever']:.4f} m")
        print(f"Average minimum distance: {safety_stats['average_minimum_distance']:.4f} m")
        print(f"Total safety violations: {safety_stats['safety_margin_violations']}")
        print(f"Collision rate: {safety_stats['collision_rate']*100:.2f}%")

if __name__ == "__main__":
    try:
        # Run simulation
        sim = SimulationEnvironment()
        sim.run_simulation()
        sim.analyze_results()
    except Exception as e:
        logging.error(f"Failed to run simulation: {str(e)}")
        raise