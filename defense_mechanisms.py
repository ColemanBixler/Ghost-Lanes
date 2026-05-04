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
                 temporal_threshold: float = 2.0, 
                 width_range: Tuple[float, float] = (100, 200),
                 min_confidence: float = 0.40):
        
        self.history = deque(maxlen=history_size)
        self.temporal_threshold = temporal_threshold
        self.width_range = width_range
        self.min_confidence = min_confidence
        self.metrics = DefenseMetrics(0, 0, 0, 0)
        self.output_dir = Path("ghost_lanes_results")

    def validate_detection(self, detection: LaneDetection, is_real_lane: bool = True) -> Tuple[bool, Dict]:
        results = {
            "confidence_pass": detection.confidence >= self.min_confidence,
            "temporal_pass": self._check_temporal_consistency(detection),
            "geometric_pass": self._check_geometric_consistency(detection)
        }
        
        is_valid = all(results.values())
        
        if is_real_lane:
            if is_valid: self.metrics.true_positives += 1
            else: self.metrics.false_negatives += 1
        else:
            if is_valid: self.metrics.false_positives += 1
            else: self.metrics.true_negatives += 1
            
        self.history.append(detection)
        return is_valid, results

    def _check_temporal_consistency(self, detection: LaneDetection) -> bool:
        if not self.history: return True
        prev = self.history[-1]
        if detection.left_lane is not None and prev.left_lane is not None:
            min_len = min(len(detection.left_lane), len(prev.left_lane))
            dist = np.mean(np.linalg.norm(detection.left_lane[:min_len] - prev.left_lane[:min_len], axis=1))
            return dist < self.temporal_threshold
        return True

    def _check_geometric_consistency(self, detection: LaneDetection) -> bool:
        if detection.left_lane is None or detection.right_lane is None:
            return False
        width = np.abs(detection.right_lane[0][0] - detection.left_lane[0][0])
        return self.width_range[0] <= width <= self.width_range[1]

    def generate_summary_report(self) -> str:
        """Generates a formatted report and saves it to a file"""
        p = self.metrics.precision
        r = self.metrics.recall
        f1 = 2 * (p * r) / (p + r + 1e-6)

        report = [
            "="*70,
            "GHOST LANES: FINAL DEFENSE PERFORMANCE REPORT",
            "="*70,
            f"Precision: {p:.4f} (Ability to reject ghosts)",
            f"Recall:    {r:.4f} (Ability to keep real lanes)",
            f"F1 Score:  {f1:.4f}",
            "-" * 30,
            "RAW COUNTS:",
            f"  True Positives (Real lanes kept): {self.metrics.true_positives}",
            f"  True Negatives (Ghosts blocked):  {self.metrics.true_negatives}",
            f"  False Positives (Ghosts leaked):  {self.metrics.false_positives}",
            f"  False Negatives (Real lanes cut): {self.metrics.false_negatives}",
            "="*70
        ]
        
        report_text = "\n".join(report)
        
        # Save to file
        self.output_dir.mkdir(exist_ok=True)
        with open(self.output_dir / 'defense_summary.txt', 'w') as f:
            f.write(report_text)
            
        return report_text

def run_defense_evaluation():
    print("="*60)
    print("GHOST LANES: DEFENSE EVALUATION")
    print("="*60)
    
    defense = GhostLanesDefense()
    results_dir = Path("ghost_lanes_results")
    attack_files = list(results_dir.glob("metrics_*.json"))
    
    if not attack_files:
        print(f"ERROR: No metrics files found in {results_dir}")
        return

    for file_path in attack_files:
        print(f"Processing: {file_path.name}...")
        with open(file_path, 'r') as f:
            data = json.load(f)

        deviations = data.get('lateral_deviations', [])
        for i, dev in enumerate(deviations):
            is_attack_active = abs(dev) > 0.5 
            
            det = LaneDetection(
                left_lane=np.array([[100 + dev, 200]]),
                right_lane=np.array([[300 + dev, 200]]),
                confidence=data.get('confidence', 0.85),
                timestamp=float(i) / 10.0
            )
            defense.validate_detection(det, is_real_lane=not is_attack_active)

    # Generate and print report
    final_report = defense.generate_summary_report()
    print(final_report)

if __name__ == "__main__":
    run_defense_evaluation()