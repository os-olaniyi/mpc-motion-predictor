from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from collision_avoidance import (
    Obstacle
    )

from mpc import (
    RobotState
    )

@dataclass
class CollisionMetrics:
    """Store collision-related metrics"""
    min_distance: float
    collision_occurred: bool
    closest_obstacle: Optional[Obstacle]
    time_to_collision: Optional[float]
    collision_point: Optional[Tuple[float, float]]

class CollisionDetector:
    """Handles collision detection and metrics"""
    def __init__(self, obstacles: List[Obstacle], safety_distance: float):
        self.obstacles = obstacles
        self.safety_distance = safety_distance
        self.metrics_history = []
        self.collision_events = []
        
    def check_collision(self, state: RobotState) -> CollisionMetrics:
        """Check for collisions and compute metrics"""
        min_distance = float('inf')
        closest_obstacle = None
        collision_occurred = False
        collision_point = None
        time_to_collision = None
        
        # Check distance to each obstacle
        for obstacle in self.obstacles:
            distance = np.sqrt(
                (state.x - obstacle.x)**2 + 
                (state.y - obstacle.y)**2
            ) - obstacle.radius
            
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle
            
            if distance <= 0:  # Collision occurred
                collision_occurred = True
                collision_point = (state.x, state.y)
        
        metrics = CollisionMetrics(
            min_distance=min_distance,
            collision_occurred=collision_occurred,
            closest_obstacle=closest_obstacle,
            time_to_collision=time_to_collision,
            collision_point=collision_point
        )
        
        self.metrics_history.append(metrics)
        if collision_occurred:
            self.collision_events.append(metrics)
        
        return metrics
        
    def predict_time_to_collision(self, 
                                state: RobotState, 
                                predicted_trajectory: np.ndarray,
                                dt: float) -> Optional[float]:
        """Predict time to collision based on predicted trajectory"""
        for i, point in enumerate(predicted_trajectory):
            for obstacle in self.obstacles:
                distance = np.sqrt(
                    (point[0] - obstacle.x)**2 + 
                    (point[1] - obstacle.y)**2
                ) - obstacle.radius
                
                if distance <= 0:
                    return i * dt
        return None
    
    def get_safety_statistics(self) -> Dict:
        """Compute safety-related statistics"""
        min_distances = [m.min_distance for m in self.metrics_history]
        
        return {
            'minimum_distance_ever': min(min_distances),
            'average_minimum_distance': np.mean(min_distances),
            'total_collisions': len(self.collision_events),
            'collision_rate': len(self.collision_events) / len(self.metrics_history) if self.metrics_history else 0,
            'safety_margin_violations': sum(1 for d in min_distances if d < self.safety_distance),
            'average_safety_margin': np.mean([max(0, d - self.safety_distance) for d in min_distances])
        }
    
    def plot_safety_metrics(self, save_path: Optional[Path] = None):
        """Generate plots of safety metrics"""
        if not self.metrics_history:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Minimum distance over time
        distances = [m.min_distance for m in self.metrics_history]
        times = range(len(distances))
        ax1.plot(times, distances, 'b-', label='Minimum Distance')
        ax1.axhline(y=self.safety_distance, color='r', linestyle='--', 
                   label=f'Safety Distance ({self.safety_distance}m)')
        ax1.set_title('Minimum Distance to Obstacles')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Distance (m)')
        ax1.grid(True)
        ax1.legend()
        
        # Distance histogram
        ax2.hist(distances, bins=30, density=True, alpha=0.7)
        ax2.axvline(self.safety_distance, color='r', linestyle='--', 
                   label=f'Safety Distance ({self.safety_distance}m)')
        ax2.set_title('Distribution of Minimum Distances')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Density')
        ax2.grid(True)
        ax2.legend()
        
        # Safety margin violations
        margins = [max(0, d - self.safety_distance) for d in distances]
        ax3.plot(times, margins, 'g-', label='Safety Margin')
        ax3.fill_between(times, margins, alpha=0.3)
        ax3.set_title('Safety Margin Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Margin (m)')
        ax3.grid(True)
        ax3.legend()
        
        # Collision events
        if self.collision_events:
            collision_times = [self.metrics_history.index(event) 
                             for event in self.collision_events]
            collision_dists = [0 for _ in collision_times]  # Plot at y=0
            ax4.scatter(collision_times, collision_dists, 
                       color='red', marker='x', s=100, label='Collisions')
        ax4.plot(times, distances, 'b-', alpha=0.5)
        ax4.set_title('Collision Events')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Distance at Collision (m)')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def generate_safety_report(self, save_path: Optional[Path] = None) -> str:
        """Generate a detailed safety report"""
        stats = self.get_safety_statistics()
        
        report = [
            "Safety Analysis Report",
            "=====================",
            f"Total Time Steps: {len(self.metrics_history)}",
            f"Total Collisions: {stats['total_collisions']}",
            f"Collision Rate: {stats['collision_rate']*100:.2f}%",
            "",
            "Distance Metrics",
            "----------------",
            f"Minimum Distance Ever: {stats['minimum_distance_ever']:.3f} m",
            f"Average Minimum Distance: {stats['average_minimum_distance']:.3f} m",
            f"Safety Margin Violations: {stats['safety_margin_violations']}",
            f"Average Safety Margin: {stats['average_safety_margin']:.3f} m",
            "",
            "Collision Events",
            "----------------"
        ]
        
        for i, event in enumerate(self.collision_events, 1):
            report.extend([
                f"Collision {i}:",
                f"  Distance: {event.min_distance:.3f} m",
                f"  Position: {event.collision_point}",
                f"  Obstacle: ({event.closest_obstacle.x}, {event.closest_obstacle.y})"
            ])
        
        report_text = "\n".join(report)
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text

class SafetyMonitor:
    """Monitors and analyzes safety-related performance"""
    def __init__(self, collision_detector: CollisionDetector):
        self.collision_detector = collision_detector
        self.start_time = time.time()
        self.safety_violations = []
        self.near_misses = []
        self.reaction_times = []
        
    def update(self, 
              current_state: RobotState,
              predicted_trajectory: Optional[np.ndarray] = None,
              control_input: Optional[np.ndarray] = None) -> Dict:
        """Update safety monitoring with current state"""
        metrics = self.collision_detector.check_collision(current_state)
        
        # Check for safety violations
        if metrics.min_distance < self.collision_detector.safety_distance:
            self.safety_violations.append({
                'time': time.time() - self.start_time,
                'distance': metrics.min_distance,
                'state': current_state.to_array(),
                'control': control_input.tolist() if control_input is not None else None
            })
        
        # Check for near misses (within 120% of safety distance)
        if (metrics.min_distance < self.collision_detector.safety_distance * 1.2 and
            metrics.min_distance > self.collision_detector.safety_distance):
            self.near_misses.append({
                'time': time.time() - self.start_time,
                'distance': metrics.min_distance,
                'state': current_state.to_array()
            })
        
        # Predict potential collisions
        if predicted_trajectory is not None:
            time_to_collision = self.collision_detector.predict_time_to_collision(
                current_state, predicted_trajectory, 0.1  # assuming dt = 0.1
            )
            if time_to_collision is not None:
                self.reaction_times.append(time_to_collision)
        
        return {
            'current_distance': metrics.min_distance,
            'safety_violations': len(self.safety_violations),
            'near_misses': len(self.near_misses),
            'avg_reaction_time': np.mean(self.reaction_times) if self.reaction_times else None
        }
    
    def get_safety_metrics(self) -> Dict:
        """Get comprehensive safety metrics"""
        return {
            'safety_violations': {
                'count': len(self.safety_violations),
                'avg_distance': np.mean([v['distance'] for v in self.safety_violations]) if self.safety_violations else None,
                'min_distance': min([v['distance'] for v in self.safety_violations]) if self.safety_violations else None
            },
            'near_misses': {
                'count': len(self.near_misses),
                'avg_distance': np.mean([n['distance'] for n in self.near_misses]) if self.near_misses else None
            },
            'reaction_time': {
                'avg': np.mean(self.reaction_times) if self.reaction_times else None,
                'min': min(self.reaction_times) if self.reaction_times else None,
                'max': max(self.reaction_times) if self.reaction_times else None
            }
        }

class SafetyPerformanceAnalyzer:
    """Analyzes and reports on overall safety performance"""
    def __init__(self, safety_monitor: SafetyMonitor):
        self.safety_monitor = safety_monitor
        self.performance_history = []
        
    def analyze_performance(self) -> Dict:
        """Analyze overall safety performance"""
        metrics = self.safety_monitor.get_safety_metrics()
        
        # Calculate performance scores
        safety_score = self._calculate_safety_score(metrics)
        reaction_score = self._calculate_reaction_score(metrics)
        overall_score = (safety_score + reaction_score) / 2
        
        analysis = {
            'overall_safety_score': overall_score,
            'safety_score': safety_score,
            'reaction_score': reaction_score,
            'risk_level': self._determine_risk_level(overall_score),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        self.performance_history.append(analysis)
        return analysis
    
    def _calculate_safety_score(self, metrics: Dict) -> float:
        """Calculate safety score based on violations and near misses"""
        base_score = 100
        
        # Deduct points for safety violations
        if metrics['safety_violations']['count'] > 0:
            violation_penalty = min(50, metrics['safety_violations']['count'] * 10)
            base_score -= violation_penalty
        
        # Deduct points for near misses
        if metrics['near_misses']['count'] > 0:
            near_miss_penalty = min(30, metrics['near_misses']['count'] * 5)
            base_score -= near_miss_penalty
        
        return max(0, base_score)
    
    def _calculate_reaction_score(self, metrics: Dict) -> float:
        """Calculate score based on reaction times"""
        if not metrics['reaction_time']['avg']:
            return 100
        
        # Assume ideal reaction time is 0.5 seconds
        avg_reaction_time = metrics['reaction_time']['avg']
        score = 100 * (0.5 / max(avg_reaction_time, 0.5))
        return min(100, score)
    
    def _determine_risk_level(self, overall_score: float) -> str:
        """Determine risk level based on overall score"""
        if overall_score >= 90:
            return "Low Risk"
        elif overall_score >= 70:
            return "Moderate Risk"
        elif overall_score >= 50:
            return "High Risk"
        else:
            return "Critical Risk"
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate safety recommendations based on metrics"""
        recommendations = []
        
        if metrics['safety_violations']['count'] > 0:
            recommendations.append(
                "Increase safety distance or adjust obstacle avoidance weights"
            )
        
        if metrics['near_misses']['count'] > metrics['safety_violations']['count']:
            recommendations.append(
                "Consider implementing more conservative approach to obstacles"
            )
        
        if metrics['reaction_time']['avg'] and metrics['reaction_time']['avg'] > 1.0:
            recommendations.append(
                "Optimize prediction horizon or control parameters for faster response"
            )
        
        return recommendations if recommendations else ["No specific recommendations at this time"]