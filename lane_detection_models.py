"""
Ghost Lanes: Lane Detection Model Integration
Supports SCNN lane detection model
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LaneDetectionModel(ABC):
    """
    Abstract base class for lane detection models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize lane detection model
        
        Args:
            model_path: Path to pretrained model weights
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def detect_lanes(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect lanes in image
        
        Returns:
            Dictionary with 'left_lane', 'right_lane', 'confidence'
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: torch.Tensor) -> Dict[str, np.ndarray]:
        """Postprocess model output to lane coordinates"""
        pass


class SCNNModel(LaneDetectionModel):
    """
    Spatial CNN (SCNN) model for lane detection
    Reference: Pan et al. "Spatial As Deep: Spatial CNN for Traffic Scene Understanding"
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (800, 288)):
        """
        Initialize SCNN model
        
        Args:
            model_path: Path to pretrained weights
            input_size: Model input size (width, height)
        """
        super().__init__(model_path)
        self.input_size = input_size
        
        # Build model architecture
        self.model = self._build_scnn()
        
        # Load pretrained weights if available
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded SCNN weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _build_scnn(self) -> nn.Module:
        """
        Build SCNN architecture
        Simplified version for demonstration
        """
        class SCNN(nn.Module):
            def __init__(self):
                super(SCNN, self).__init__()
                
                # VGG-16 backbone (simplified)
                self.backbone = nn.Sequential(
                    # Conv1
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    # Conv2
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    # Conv3
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                )
                
                # Spatial CNN layers (message passing)
                self.scnn = nn.ModuleList([
                    nn.Conv2d(256, 256, (1, 9), padding=(0, 4)),  # Horizontal
                    nn.Conv2d(256, 256, (9, 1), padding=(4, 0)),  # Vertical
                ])
                
                # Lane prediction head
                self.lane_head = nn.Sequential(
                    nn.Conv2d(256, 128, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 5, 1),  # 4 lanes + background
                )
            
            def forward(self, x):
                # Backbone
                features = self.backbone(x)
                
                # SCNN message passing
                # Horizontal pass (left-to-right, right-to-left)
                for i in range(features.shape[3]):
                    features[:, :, :, i] = self.scnn[0](features[:, :, :, max(0, i-4):min(features.shape[3], i+5)])[:, :, :, 4]
                
                # Vertical pass (top-to-bottom, bottom-to-top)
                for i in range(features.shape[2]):
                    features[:, :, i, :] = self.scnn[1](features[:, :, max(0, i-4):min(features.shape[2], i+5), :])[:, :, 4, :]
                
                # Lane prediction
                lanes = self.lane_head(features)
                
                return lanes
        
        return SCNN()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for SCNN
        
        Args:
            image: Input image (H x W x 3)
            
        Returns:
            Preprocessed tensor (1 x 3 x H x W)
        """
        # Resize
        img = cv2.resize(image, self.input_size)
        
        # Normalize (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor (C x H x W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
    
    def detect_lanes(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect lanes using SCNN
        
        Args:
            image: Input image (H x W x 3)
            
        Returns:
            Dictionary with detected lane information
        """
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        lanes = self.postprocess(output, image.shape[:2])
        
        return lanes
    
    def postprocess(self, 
                    model_output: torch.Tensor,
                    original_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Postprocess SCNN output to lane coordinates
        
        Args:
            model_output: Model output tensor (1 x 5 x H x W)
            original_size: Original image size (H, W)
            
        Returns:
            Dictionary with left_lane, right_lane, confidence
        """
        # Get predicted lanes (argmax)
        lanes_pred = torch.argmax(model_output, dim=1).squeeze().cpu().numpy()
        
        # Resize to original image size
        lanes_pred = cv2.resize(
            lanes_pred.astype(np.uint8), 
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Extract individual lanes
        left_lane = self._extract_lane_points(lanes_pred, lane_id=1)
        right_lane = self._extract_lane_points(lanes_pred, lane_id=2)
        
        # Calculate confidence (percentage of pixels detected)
        total_pixels = lanes_pred.size
        lane_pixels = np.sum((lanes_pred > 0) & (lanes_pred < 5))
        confidence = lane_pixels / total_pixels
        
        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'confidence': confidence
        }
    
    def _extract_lane_points(self, 
                             lane_mask: np.ndarray, 
                             lane_id: int) -> Optional[np.ndarray]:
        """
        Extract lane points from segmentation mask
        
        Args:
            lane_mask: Lane segmentation mask (H x W)
            lane_id: Lane ID to extract
            
        Returns:
            Array of lane points (N x 2) or None
        """
        # Get lane pixels
        lane_pixels = np.where(lane_mask == lane_id)
        
        if len(lane_pixels[0]) == 0:
            return None
        
        # Group by rows and get average column for each row
        points = []
        for y in range(lane_mask.shape[0]):
            row_pixels = lane_pixels[1][lane_pixels[0] == y]
            if len(row_pixels) > 0:
                x = np.mean(row_pixels)
                points.append([x, y])
        
        if len(points) < 2:
            return None
        
        return np.array(points)


class ModelFactory:
    """Factory for creating lane detection models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> LaneDetectionModel:
        """
        Create lane detection model
        
        Args:
            model_type: Type of model ('scnn')
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Lane detection model instance
        """
        if model_type.lower() == 'scnn':
            return SCNNModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def test_models():
    """Test lane detection models with synthetic data"""
    print("Testing Lane Detection Models\n")
    
    # Create synthetic test image
    test_image = np.zeros((320, 480, 3), dtype=np.uint8)
    
    # Draw simple lane markings
    cv2.line(test_image, (160, 310), (220, 100), (255, 255, 255), 12)
    cv2.line(test_image, (380, 310), (280, 100), (255, 255, 255), 1)
    
    # Test SCNN
    print("Testing SCNN model...")
    scnn = ModelFactory.create_model('scnn')
    scnn_result = scnn.detect_lanes(test_image)
    print(f"  Left lane detected: {scnn_result['left_lane'] is not None}")
    print(f"  Right lane detected: {scnn_result['right_lane'] is not None}")
    print(f"  Confidence: {scnn_result['confidence']:.3f}\n")


if __name__ == "__main__":
    test_models()
