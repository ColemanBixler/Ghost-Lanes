# Ghost Lanes: Lane Injection Attacks on Autonomous Vehicles

Implementation and evaluation of lane injection attacks in the CARLA simulator, specifically optimized for SCNN.

## Overview

This research framework evaluates the vulnerability of deep-learning-based perception systems to adversarial lane markings.
- **Parallel Attack**: Injects a fake lane at a constant lateral offset.
- **Convergent Attack**: Guides the vehicle toward the center or opposite lane.
- **Divergent Attack**: Forces the vehicle toward the road edge/shoulder.

## Key Updates for M1+ Mac / Rosetta Environments
The current version has been stabilized for ARM-based architectures running CARLA via Rosetta:
- **Resolution Scaling**: Calibrated for 480x320 synthetic injection to maintain FPS.
- **Model Stability**: Optimized SCNN spatial aggregation to handle emulation latency.
- **Idempotent Cleanup**: Improved actor destruction logic to prevent `std::runtime_error` crashes during simulation resets.

## Architecture

- `ghost_lanes_attack.py`: Core simulator and attack orchestrator.
- `lane_detection_models.py`: Model definitions for SCNN and Ultra-Fast.
- `defense_mechanisms.py`: Implementation of temporal and geometric sanity checks.
- `attack_metrics.py`: Automated report generator for ASR and Lateral Deviation.

## Setup Information
- Use [CARLA 0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13) or newer.
- Use Python 3.8 or newer and install missing dependencies with `pip install -r requirements.txt`

### 1. Start CARLA Simulator
Execute the [CARLA simulator app](https://github.com/carla-simulator/carla) on your machine, and check the IP address that CARLA is connecting to. 
- If you are not on Mac it should be localhost. 
- Otherwise, run `ifconfig | grep -A 3 "en" | grep "inet "` to and make sure to replace `localhost` with the given IP address after inet.

### 2. Run the Attack Simulation
Execute the primary attack suite. This will generate individual metrics files for each attack geometry.
```bash
python3 ghost_lanes_attack.py
```
*Note: This script now handles actor cleanup safely to allow back-to-back testing of Parallel, Convergent, and Divergent types.*

### 3. Generate Research Metrics
Process the raw JSON data in `ghost_lanes_results/` to generate visualization plots and the final summary report.
```bash˚v
python3 attack_metrics.py
```
Outputs: `attack_analysis.png`, `summary_report.txt`

### 4. Evaluate Defense Performance
Run the defense validator against the captured attack logs to calculate Precision, Recall, and F1 scores.
```bash
python3 defense_mechanisms.py
```

## Research Metrics

The framework tracks four primary Safety-Critical Metrics:
1. **Max Lateral Deviation (MLD)**: The maximum distance (meters) the vehicle was pulled from the true lane center.
2. **Attack Success Rate (ASR)**: Percentage of the trajectory where the vehicle deviated beyond 0.8m.
3. **LDW Evasion Rate**: Ratio of successful attacks that did not trigger the Lane Departure Warning threshold.
4. **Drift Velocity**: The rate of lateral shift (m/s) used to evade temporal filters.

## Defense Strategy

The project evaluates a three-stage defense pipeline:
1. **Temporal Filtering**: Compares current detection against a 5-frame history (Optimal threshold: 2.0).
2. **Geometric Consistency**: Validates that lane width remains within expected bounds (100-200px for 480p).
3. **Confidence Monitoring**: Rejects low-confidence detections often associated with noisy adversarial inputs.

## Results Summary

Current testing in `Town01` reveals that while simple temporal filters can catch the initial injection "snap," smooth lateral drifts (as seen in Divergent attacks) frequently bypass basic geometric consistency checks, leading to deviations exceeding **11.0 meters**.

## Citation
```
Ghost Lanes: Lane Injection Attacks on Autonomous Vehicles
Coleman Bixler, Vikash Balaji Kokku
University of Oklahoma, 2026
```
