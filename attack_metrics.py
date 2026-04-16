"""
Ghost Lanes: Metrics Collection and Evaluation
Tracks ASR, MLD, DOT, and LDW evasion metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from types import SimpleNamespace
from dataclasses import dataclass, fields
import json
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class AttackMetrics:
    """Container for attack evaluation metrics"""
    attack_success_rate: float = 0.0
    max_lateral_deviation: float = 0.0
    lateral_deviations: List[float] = None
    ldw_evasion_rate: float = 0.0
    average_lateral_deviation: float = 0.0
    drift_velocity: float = 0.0
    time_to_max_deviation: float = 0.0
    attack_type: str = "unknown"
    offset: float = 0.0
    length: float = 0.0
    opacity: float = 0.0

class ExperimentRunner:
    """
    Runs complete experimental setup and processes results
    """
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[AttackMetrics] = []

    def load_json_results(self, folder_path: str):
        """
        Properly reconstructs AttackMetrics objects from saved JSON files
        """
        path = Path(folder_path)
        files = list(path.glob("metrics_*.json"))
        
        for file_path in files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Calculate missing metrics required for plotting
                deviations = data.get('lateral_deviations', [])
                mld = data.get('max_lateral_deviation', 0.0)
                
                # Logic-based metric injection
                if 'attack_success_rate' not in data:
                    # ASR: Percentage of frames > 0.1m deviation
                    data['attack_success_rate'] = (sum(1 for d in deviations if abs(d) >= 0.1) / len(deviations) * 100.0) if deviations else 0.0
                
                if 'ldw_evasion_rate' not in data:
                    # Evasion: Percentage of frames < 0.3m (LDW threshold)
                    data['ldw_evasion_rate'] = (sum(1 for d in deviations if abs(d) < 0.3) / len(deviations) * 100.0) if deviations else 0.0
                
                if 'drift_velocity' not in data:
                    # Average change in deviation per frame (assumed 10 FPS)
                    data['drift_velocity'] = np.mean(np.abs(np.diff(deviations))) * 10.0 if len(deviations) > 1 else 0.0

                # Filter data to only include valid AttackMetrics fields
                valid_fields = {f.name for f in fields(AttackMetrics)}
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                
                metrics_obj = AttackMetrics(**filtered_data)
                self.results.append(metrics_obj)
                print(f"Successfully loaded: {file_path.name}")

    def generate_plots(self):
        """Generate visualization plots from results"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ghost Lanes Attack Analysis', fontsize=30, fontweight='bold')
        
        attack_types = ['parallel', 'convergent', 'divergent']
        
        # Helper to group data
        def get_avg(attr):
            return [np.mean([getattr(m, attr) for m in self.results if m.attack_type == t]) 
                    if any(m.attack_type == t for m in self.results) else 0 for t in attack_types]

        # Plotting
        asr_bars = axes[0, 0].bar(attack_types, get_avg('attack_success_rate'), color=['#1E2761', '#2C5F2D', '#C72C31'])
        axes[0, 0].bar_label(asr_bars, fmt='%.2f%%', label_type='center', color='white', fontsize=18)
        axes[0, 0].set_title('ASR by Attack Type (%)', fontsize=22, fontweight='bold')
        
        mld_bars = axes[0, 1].bar(attack_types, get_avg('max_lateral_deviation'), color=['#1E2761', '#2C5F2D', '#C72C31'])
        axes[0, 1].bar_label(mld_bars, fmt='%.2f m', label_type='center', color='white', fontsize=18)
        axes[0, 1].set_title('MLD by Attack Type (m)', fontsize=22, fontweight='bold')
        
        ldw_bars = axes[1, 0].bar(attack_types, get_avg('ldw_evasion_rate'), color=['#1E2761', '#2C5F2D', '#C72C31'])
        axes[1, 0].bar_label(ldw_bars, fmt='%.2f%%', label_type='center', color='white', fontsize=18)
        axes[1, 0].set_title('LDW Evasion Rate (%)', fontsize=22, fontweight='bold')
        
        dv_bars = axes[1, 1].bar(attack_types, get_avg('drift_velocity'), color=['#1E2761', '#2C5F2D', '#C72C31'])
        axes[1, 1].bar_label(dv_bars, fmt='%.2f%%', label_type='center', color='white', fontsize=18)
        axes[1, 1].set_title('Avg Drift Velocity (m/s)', fontsize=22, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'attack_analysis.png', dpi=300)
        print(f"Plot saved to: {self.output_dir / 'attack_analysis.png'}")

    def generate_summary_report(self) -> str:
        if not self.results: return "No results"
        
        report = ["="*70, "GHOST LANES EXPERIMENT SUMMARY", "="*70]
        for atype in sorted(set(m.attack_type for m in self.results)):
            subset = [m for m in self.results if m.attack_type == atype]
            report.append(f"\nTYPE: {atype.upper()}")
            report.append(f"  Mean MLD: {np.mean([m.max_lateral_deviation for m in subset]):.3f}m")
            report.append(f"  Mean ASR: {np.mean([m.attack_success_rate for m in subset]):.2f}%")
        
        report_text = "\n".join(report)
        with open(self.output_dir / 'summary_report.txt', 'w') as f: f.write(report_text)
        return report_text

def main():
    output_dir = "ghost_lanes_results"
    runner = ExperimentRunner(output_dir=output_dir)
    
    # Use the new robust loader
    runner.load_json_results(output_dir)
    
    if runner.results:
        runner.generate_plots()
        print(runner.generate_summary_report())

if __name__ == "__main__":
    main()