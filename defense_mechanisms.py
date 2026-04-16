"""
Ghost Lanes: Defense Mechanisms - M1 Mac Stabilized Version
Implements temporal filtering, geometric consistency, and confidence monitoring
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from dataclasses import dataclass
import cv2
import json
from pathlib import Path

@dataclass
class LaneDetection:
    """Container for lane detection results"""
    left_lane: Optional[np.ndarray]
    right_lane: Optional[np.ndarray]
    confidence: float
    timestamp: float
    
@dataclass
class DefenseMetrics:
    """Metrics for defense evaluation"""
    true_positives: int  # Real lanes accepted
    false_positives: int # Fake lanes accepted
    true_negatives: int  # Fake lanes rejected
    false_negatives: int # Real lanes rejected
    
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

class GhostLanesDefense:
    """
    Multi-stage defense against lane injection attacks
    """
    def __init__(self, 
                 history_size: int = 5,
                 # Thresholds calibrated for 480x320 resolution
                 temporal_threshold: float = 2.0, 
                 width_range: Tuple[float, float] = (100, 200),
                 min_confidence: float = 0.40):
        
        self.history = deque(maxlen=history_size)
        self.temporal_threshold = temporal_threshold
        self.width_range = width_range
        self.min_confidence = min_confidence
        self.metrics = DefenseMetrics(0, 0, 0, 0)

    def validate_detection(self, detection: LaneDetection, is_real_lane: bool = True) -> Tuple[bool, Dict]:
        """
        Validate a detection using three layers of security
        """
        results = {
            "confidence_pass": detection.confidence >= self.min_confidence,
            "temporal_pass": self._check_temporal_consistency(detection),
            "geometric_pass": self._check_geometric_consistency(detection)
        }
        
        is_valid = all(results.values())
        
        # Update academic metrics
        if is_real_lane:
            if is_valid: self.metrics.true_positives += 1
            else: self.metrics.false_negatives += 1
        else:
            if is_valid: self.metrics.false_positives += 1
            else: self.metrics.true_negatives += 1
            
        self.history.append(detection)
        return is_valid, results

    def _check_temporal_consistency(self, detection: LaneDetection) -> bool:
        """Checks if the new lane has jumped too far from the previous frame"""
        if not self.history: return True
        prev = self.history[-1]
        
        # Calculate mean Euclidean distance between lane points
        if detection.left_lane is not None and prev.left_lane is not None:
            # Match lengths for comparison
            min_len = min(len(detection.left_lane), len(prev.left_lane))
            dist = np.mean(np.linalg.norm(detection.left_lane[:min_len] - prev.left_lane[:min_len], axis=1))
            return dist < self.temporal_threshold
        return True

    def _check_geometric_consistency(self, detection: LaneDetection) -> bool:
        """Checks if lane width is physically plausible at 480x320 resolution"""
        if detection.left_lane is None or detection.right_lane is None:
            return False
            
        # Check average horizontal distance between lanes (width)
        # Using the bottom of the image where lanes are widest
        width = np.abs(detection.right_lane[0][0] - detection.left_lane[0][0])
        return self.width_range[0] <= width <= self.width_range[1]

def run_defense_evaluation():
    """
    Main evaluation loop: Processes individual attack JSON files
    """
    print("="*60)
    print("GHOST LANES: DEFENSE EVALUATION")
    print("="*60)
    
    defense = GhostLanesDefense()
    results_dir = Path("ghost_lanes_results")
    
    # Find all individual attack metrics files
    attack_files = list(results_dir.glob("metrics_*.json"))
    
    if not attack_files:
        print(f"ERROR: No metrics files found in {results_dir}")
        return

    for file_path in attack_files:
        print(f"\nEvaluating Defense against: {file_path.name}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        # In your current attack setup, the injection usually starts after 
        # the baseline is established. 
        # We simulate the frame stream from the lateral_deviations list.
        deviations = data.get('lateral_deviations', [])
        
        for i, dev in enumerate(deviations):
            # We must simulate LaneDetection objects for the defense logic.
            # Real lanes are usually at +/- ~1.75m from center.
            # A ghost lane shifts that detection by the 'dev' value.
            
            # Assume an attack is active if deviation is significant
            is_attack_active = abs(dev) > 0.5 
            
            # Create a mock detection based on the deviation recorded
            det = LaneDetection(
                left_lane=np.array([[100 + dev, 200]]), # Mock coordinates
                right_lane=np.array([[300 + dev, 200]]),
                confidence=data.get('confidence', 0.85),
                timestamp=float(i) / 10.0 # Assuming 10 FPS
            )
            
            # Validate: is_real_lane is False if the attack has veered the car
            is_valid, reasons = defense.validate_detection(det, is_real_lane=not is_attack_active)

    print("\n" + "="*60)
    print("FINAL DEFENSE PERFORMANCE REPORT")
    print("-" * 30)
    print(f"Precision: {defense.metrics.precision:.4f} (Ability to reject ghosts)")
    print(f"Recall:    {defense.metrics.recall:.4f} (Ability to keep real lanes)")
    
    # Calculate F1 Score
    p = defense.metrics.precision
    r = defense.metrics.recall
    f1 = 2 * (p * r) / (p + r + 1e-6)
    print(f"F1 Score:  {f1:.4f}")
    
    # Summary of raw counts
    print(f"\nTrue Positives (Real lanes kept): {defense.metrics.true_positives}")
    print(f"True Negatives (Ghosts blocked):  {defense.metrics.true_negatives}")
    print(f"False Positives (Ghosts leaked):  {defense.metrics.false_positives}")
    print(f"False Negatives (Real lanes cut): {defense.metrics.false_negatives}")
    print("="*60)

if __name__ == "__main__":
    run_defense_evaluation()