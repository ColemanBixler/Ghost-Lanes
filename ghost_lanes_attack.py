#!/usr/bin/env python3
"""
Ghost Lanes: Lane Injection Attacks on Autonomous Vehicles
CARLA Simulation Framework - M1 Mac Stabilized Version
"""

import carla
import json
from pathlib import Path
import numpy as np
import cv2
import time
import random
from enum import Enum
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of lane injection attacks"""
    PARALLEL = "parallel"
    CONVERGENT = "convergent"
    DIVERGENT = "divergent"


class WeatherCondition(Enum):
    """Weather conditions for testing"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    WET = "wet"
    WET_CLOUDY = "wet_cloudy"
    SOFT_RAIN = "soft_rain"
    MID_RAIN = "mid_rain"
    HARD_RAIN = "hard_rain"


class LaneInjectionAttack:
    """
    Base class for lane injection attacks in CARLA
    """
    
    def __init__(self, 
                 attack_type: AttackType,
                 offset: float = 0.5,  # meters
                 length: float = 50.0,  # meters
                 opacity: float = 0.8,
                 width: float = 0.15):  # meters
        """
        Initialize lane injection attack
        """
        self.attack_type = attack_type
        self.offset = offset
        self.length = length
        self.opacity = opacity
        self.width = width
        
    def generate_fake_lane_points(self, 
                                   real_lane_points: np.ndarray,
                                   distance_ahead: float = 30.0) -> np.ndarray:
        """
        Generate fake lane points based on attack type
        """
        if self.attack_type == AttackType.PARALLEL:
            return self._generate_parallel_lane(real_lane_points, distance_ahead)
        elif self.attack_type == AttackType.CONVERGENT:
            return self._generate_convergent_lane(real_lane_points, distance_ahead)
        elif self.attack_type == AttackType.DIVERGENT:
            return self._generate_divergent_lane(real_lane_points, distance_ahead)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def _generate_parallel_lane(self, 
                                 real_lane_points: np.ndarray,
                                 distance_ahead: float) -> np.ndarray:
        start_idx = self._find_start_index(real_lane_points, distance_ahead)
        fake_points = []
        for i in range(start_idx, min(start_idx + int(self.length), len(real_lane_points))):
            if i + 1 >= len(real_lane_points):
                break
            direction = real_lane_points[i + 1] - real_lane_points[i]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            perpendicular = np.array([-direction[1], direction[0], 0])
            fake_point = real_lane_points[i] + perpendicular * self.offset
            fake_points.append(fake_point)
        return np.array(fake_points)
    
    def _generate_convergent_lane(self,
                                   real_lane_points: np.ndarray,
                                   distance_ahead: float) -> np.ndarray:
        start_idx = self._find_start_index(real_lane_points, distance_ahead)
        fake_points = []
        num_points = min(int(self.length), len(real_lane_points) - start_idx)
        for i in range(num_points):
            idx = start_idx + i
            if idx + 1 >= len(real_lane_points):
                break
            progress = i / max(num_points - 1, 1)
            current_offset = self.offset * (1 - progress)
            direction = real_lane_points[idx + 1] - real_lane_points[idx]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            perpendicular = np.array([-direction[1], direction[0], 0])
            fake_point = real_lane_points[idx] + perpendicular * current_offset
            fake_points.append(fake_point)
        return np.array(fake_points)
    
    def _generate_divergent_lane(self,
                                  real_lane_points: np.ndarray,
                                  distance_ahead: float) -> np.ndarray:
        start_idx = self._find_start_index(real_lane_points, distance_ahead)
        fake_points = []
        num_points = min(int(self.length), len(real_lane_points) - start_idx)
        max_offset = self.offset * 2.0
        for i in range(num_points):
            idx = start_idx + i
            if idx + 1 >= len(real_lane_points):
                break
            progress = i / max(num_points - 1, 1)
            current_offset = self.offset + (max_offset - self.offset) * progress
            direction = real_lane_points[idx + 1] - real_lane_points[idx]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            perpendicular = np.array([-direction[1], direction[0], 0])
            fake_point = real_lane_points[idx] + perpendicular * current_offset
            fake_points.append(fake_point)
        return np.array(fake_points)
    
    def _find_start_index(self, 
                          lane_points: np.ndarray, 
                          distance_ahead: float) -> int:
        cumulative_distance = 0.0
        for i in range(len(lane_points) - 1):
            segment_length = np.linalg.norm(lane_points[i + 1] - lane_points[i])
            cumulative_distance += segment_length
            if cumulative_distance >= distance_ahead:
                return i
        return max(0, len(lane_points) - int(self.length))


class CARLAAttackSimulator:
    """
    Main simulator class - Optimized for M1 Wineskin Bridging
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 2000,
                 timeout: float = 60.0): # High timeout for Rosetta
        """
        Initialize CARLA simulator connection
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = None
        self.vehicle = None
        self.camera = None
        self.camera_image = None
        self.blueprint_library = None
        
    def setup_world(self, 
                    map_name: str = 'Town01',
                    weather: WeatherCondition = WeatherCondition.CLEAR):
        """
        Setup world with stability-focused fixed time steps
        """
        logger.info(f"Loading world: {map_name}")
        self.world = self.client.load_world(map_name)
        self.blueprint_library = self.world.get_blueprint_library()
        
        self._set_weather(weather)
        
        # VITAL M1 SETTINGS: Slow down physics to let images process
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10 FPS targets better stability
        self.world.apply_settings(settings)
        
    def _set_weather(self, weather: WeatherCondition):
        weather_params = {
            WeatherCondition.CLEAR: carla.WeatherParameters.ClearNoon,
            WeatherCondition.CLOUDY: carla.WeatherParameters.CloudyNoon,
            WeatherCondition.WET: carla.WeatherParameters.WetNoon,
            WeatherCondition.WET_CLOUDY: carla.WeatherParameters.WetCloudyNoon,
            WeatherCondition.SOFT_RAIN: carla.WeatherParameters.SoftRainNoon,
            WeatherCondition.MID_RAIN: carla.WeatherParameters.MidRainSunset,
            WeatherCondition.HARD_RAIN: carla.WeatherParameters.HardRainNoon,
        }
        self.world.set_weather(weather_params[weather])
        logger.info(f"Weather set to: {weather.value}")
    
    def spawn_vehicle(self, spawn_point: Optional[carla.Transform] = None) -> carla.Vehicle:
        bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        logger.info(f"Vehicle spawned at {spawn_point.location}")
        return self.vehicle
    
    def attach_camera(self, 
                      image_width: int = 480, # Reduced for M1
                      image_height: int = 320, # Reduced for M1
                      fov: float = 90.0) -> carla.Sensor:
        """
        Attach front-facing camera with optimized resolution
        """
        if self.vehicle is None:
            raise RuntimeError("Vehicle must be spawned first")
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))
        camera_bp.set_attribute('fov', str(fov))
        
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_camera_image(image))
        logger.info(f"Camera attached ({image_width}x{image_height})")
        
        return self.camera
    
    def _process_camera_image(self, image: carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.camera_image = array[:, :, :3]
    
    def get_lane_waypoints(self, distance_ahead: float = 100.0) -> np.ndarray:
        vehicle_location = self.vehicle.get_location()
        map_obj = self.world.get_map()
        current_waypoint = map_obj.get_waypoint(vehicle_location)
        
        waypoints = [current_waypoint]
        cumulative_distance = 0.0
        next_waypoint = current_waypoint
        
        while cumulative_distance < distance_ahead:
            next_waypoints = next_waypoint.next(1.0)
            if not next_waypoints: break
            next_waypoint = next_waypoints[0]
            waypoints.append(next_waypoint)
            cumulative_distance += 1.0
        
        return np.array([[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp in waypoints])
    
    def inject_fake_lanes_in_image(self, image: np.ndarray, fake_lane_points: np.ndarray, attack: LaneInjectionAttack) -> np.ndarray:
        if self.camera is None or self.vehicle is None:
            return image
        modified_image = image.copy()
        camera_transform = self.camera.get_transform()
        image_h, image_w = image.shape[:2]
        
        # Color scheme per attack type (BGR)
        ATTACK_COLORS = {
            AttackType.PARALLEL:   {"lane": (255, 60, 60),  "border": (160, 20, 20)},  # blue
            AttackType.CONVERGENT: {"lane": (60, 255, 60),  "border": (20, 160,  20)},  # green
            AttackType.DIVERGENT:  {"lane": (60, 60, 255),  "border": (20, 20, 160)},  # red
        }
        colors = ATTACK_COLORS[attack.attack_type]

        fov = float(self.camera.attributes.get('fov', 90.0))
        focal = image_w / (2.0 * np.tan(np.radians(fov / 2.0)))

        yaw   = np.radians(camera_transform.rotation.yaw)
        pitch = np.radians(camera_transform.rotation.pitch)
        roll  = np.radians(camera_transform.rotation.roll)

        Rz = np.array([[np.cos(yaw),  -np.sin(yaw), 0],
                    [np.sin(yaw),   np.cos(yaw), 0],
                    [0,             0,           1]])
        Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                    [0,              1, 0            ],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0,           0            ],
                    [0, np.cos(roll),-np.sin(roll) ],
                    [0, np.sin(roll), np.cos(roll) ]])
        R_world = Rz @ Ry @ Rx

        cam_loc = np.array([
            camera_transform.location.x,
            camera_transform.location.y,
            camera_transform.location.z,
        ])

        # Project all 3D points → 2D, keeping them in order
        image_points = []
        for pt in fake_lane_points:
            p = pt - cam_loc
            p_cam = R_world.T @ p
            x_cam =  p_cam[1]
            y_cam = -p_cam[2]
            z_cam =  p_cam[0]

            if z_cam <= 0.5:
                continue
            u = focal * x_cam / z_cam + image_w / 2.0
            v = focal * y_cam / z_cam + image_h / 2.0
            # Keep points slightly outside frame so dashes don't vanish at edges
            if -50 <= u <= image_w + 50 and -50 <= v <= image_h + 50:
                image_points.append((u, v))

        if len(image_points) >= 2:
            pts = np.array(image_points, dtype=np.float32)
            lane_width = max(4, int(attack.width * 80))
            self._draw_dashed_lane(modified_image, pts,
                                width=lane_width,
                                opacity=attack.opacity,
                                lane_color=colors["lane"],
                                border_color=colors["border"])

        # HUD label in matching color
        label = f"GHOST LANE: {attack.attack_type.value.upper()}"
        cv2.rectangle(modified_image, (8, 8), (len(label) * 9 + 12, 30), (0, 0, 0), -1)
        cv2.putText(modified_image, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["lane"], 1, cv2.LINE_AA)

        return modified_image   


    def _draw_dashed_lane(self, image: np.ndarray, points: np.ndarray,
                        width: int = 6, opacity: float = 0.8,
                        lane_color: tuple = (0, 255, 255),
                        border_color: tuple = (0, 120, 180),
                        dash_px: int = 30, gap_px: int = 20):
        """
        Draw dashed lane markings by resampling the projected polyline at fixed
        pixel intervals, then alternating dash / gap segments.
        """
        # 1. Resample the polyline into evenly-spaced pixels
        #    Build a cumulative arc-length table first.
        diffs = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_len = cum_len[-1]

        if total_len < 1:
            return

        # Sample at every pixel along the curve
        sample_distances = np.arange(0, total_len, 1.0)
        sampled = np.zeros((len(sample_distances), 2), dtype=np.float32)
        seg_idx = 0
        for k, d in enumerate(sample_distances):
            while seg_idx < len(seg_lengths) - 1 and cum_len[seg_idx + 1] < d:
                seg_idx += 1
            t = (d - cum_len[seg_idx]) / (seg_lengths[seg_idx] + 1e-6)
            sampled[k] = points[seg_idx] + t * diffs[seg_idx]

        # 2. Walk along the sampled points, alternating dash / gap
        overlay = image.copy()
        period = dash_px + gap_px
        in_dash = True
        seg_start = 0

        for i in range(1, len(sampled)):
            pos_in_period = i % period
            currently_in_dash = pos_in_period < dash_px

            if currently_in_dash != in_dash:
                # Transition: flush the current segment
                if in_dash and i - seg_start >= 2:
                    p0 = tuple(sampled[seg_start].astype(int))
                    p1 = tuple(sampled[i - 1].astype(int))
                    cv2.line(overlay, p0, p1, border_color, width + 3)
                    cv2.line(overlay, p0, p1, lane_color,   width)
                seg_start = i
                in_dash = currently_in_dash

        # Flush final segment
        if in_dash and len(sampled) - seg_start >= 2:
            p0 = tuple(sampled[seg_start].astype(int))
            p1 = tuple(sampled[-1].astype(int))
            cv2.line(overlay, p0, p1, border_color, width + 3)
            cv2.line(overlay, p0, p1, lane_color,   width)

        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)


    def _world_to_image(self, world_point: np.ndarray, camera_transform: carla.Transform, image_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int]]:
        image_h, image_w = image_shape[:2]
        fov = 90.0
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        camera_location = camera_transform.location
        point_vector = world_point - np.array([camera_location.x, camera_location.y, camera_location.z])
        
        if point_vector[0] <= 0: return None
        x_2d = int(focal * point_vector[1] / point_vector[0] + image_w / 2)
        y_2d = int(focal * point_vector[2] / point_vector[0] + image_h / 2)
        
        if 0 <= x_2d < image_w and 0 <= y_2d < image_h: return (x_2d, y_2d)
        return None
    
    def run_attack_experiment(self, attack: LaneInjectionAttack, duration: float = 30.0, save_video: bool = True, video_path: str = "attack_output.avi") -> Dict:
        metrics = {'attack_type': attack.attack_type.value, 'frames': 0, 'max_lateral_deviation': 0.0, 'average_lateral_deviation': 0.0, 'lateral_deviations': []}
        
        # Autopilot to see how the car reacts to fakes
        self.vehicle.set_autopilot(True)
        map_obj = self.world.get_map()
        
        # Start recording only after we have an image
        while self.camera_image is None:
            self.world.tick()
            time.sleep(0.1)

        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            h, w = self.camera_image.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                self.world.tick()
                vehicle_location = self.vehicle.get_location()
                current_waypoint = map_obj.get_waypoint(vehicle_location)
                
                lat_dev = self._calculate_lateral_deviation(vehicle_location, current_waypoint)
                metrics['lateral_deviations'].append(lat_dev)
                metrics['max_lateral_deviation'] = max(metrics['max_lateral_deviation'], abs(lat_dev))
                
                lane_points = self.get_lane_waypoints()
                fake_lane_points = attack.generate_fake_lane_points(lane_points)
                
                if self.camera_image is not None:
                    attacked_image = self.inject_fake_lanes_in_image(self.camera_image, fake_lane_points, attack)
                    if video_writer: video_writer.write(attacked_image)
                    metrics['frames'] += 1
        finally:
            if video_writer: 
                video_writer.release()
            
            if metrics['lateral_deviations']:
                metrics['average_lateral_deviation'] = float(np.mean(np.abs(metrics['lateral_deviations'])))
            
            # --- NEW: Save data for attack_metrics.py ---
            output_dir = Path("ghost_lanes_results")
            output_dir.mkdir(exist_ok=True)
            
            # We save a separate json for each attack type
            log_name = output_dir / f"metrics_{attack.attack_type.value}.json"
            with open(log_name, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Saved results to {log_name}")
            
        return metrics
    
    def _calculate_lateral_deviation(self, vehicle_location: carla.Location, lane_waypoint: carla.Waypoint) -> float:
        lane_location = lane_waypoint.transform.location
        deviation_vector = np.array([vehicle_location.x - lane_location.x, vehicle_location.y - lane_location.y])
        lane_rotation = lane_waypoint.transform.rotation
        lane_direction = np.array([np.cos(np.radians(lane_rotation.yaw)), np.sin(np.radians(lane_rotation.yaw))])
        perpendicular = np.array([-lane_direction[1], lane_direction[0]])
        return np.dot(deviation_vector, perpendicular)
    
    def cleanup(self):
        """
        Safe cleanup that checks if actors still exist
        """
        logger.info("Cleaning up CARLA actors...")
        
        # Check if the camera exists and is still valid in the world
        if self.camera and self.camera.is_alive:
            try:
                self.camera.destroy()
            except RuntimeError:
                pass # Already destroyed by CARLA
            self.camera = None
            
        # Check if the vehicle exists and is still valid
        if self.vehicle and self.vehicle.is_alive:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                pass # Already destroyed by CARLA
            self.vehicle = None
            
        # Reset world settings safely
        if self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except RuntimeError:
                pass

def main():
    simulator = CARLAAttackSimulator(host='localhost', port=2000)
    try:
        simulator.setup_world(map_name='Town01', weather=WeatherCondition.CLEAR)
        
        # Define the three attack scenarios
        attacks = [
            LaneInjectionAttack(AttackType.PARALLEL, offset=0.5, length=50.0),
            LaneInjectionAttack(AttackType.CONVERGENT, offset=0.8, length=60.0),
            LaneInjectionAttack(AttackType.DIVERGENT, offset=0.5, length=50.0),
        ]
        
        for attack in attacks:
            logger.info(f"--- STARTING ATTACK: {attack.attack_type.value} ---")
            
            # Re-spawn or reset vehicle position before each attack if needed
            simulator.spawn_vehicle() 
            simulator.attach_camera(image_width=480, image_height=320)
            
            # Stabilization
            time.sleep(5)
            for _ in range(10): simulator.world.tick()
            
            # Run the specific attack
            video_name = f"ghost_lane_{attack.attack_type.value}.avi"
            metrics = simulator.run_attack_experiment(
                attack=attack, 
                duration=20.0, 
                save_video=True, 
                video_path=video_name
            )
            
            print(f"Results for {attack.attack_type.value}: Max Dev = {metrics['max_lateral_deviation']:.3f}m")
            
            # Clean up actors before the next loop iteration
            simulator.cleanup()
            time.sleep(3) # Give the bridge a second to breathe
            
    finally:
        print("Done!")

if __name__ == "__main__":
    main()

