"""
Depth Estimation Module
Uses MiDaS for real-time monocular depth estimation
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import os


class DepthEstimator:
    """
    Real-time depth estimation using MiDaS
    """
    
    def __init__(self, model_type="DPT_Large", checkpoints_dir="checkpoints"):
        """
        Initialize depth estimator
        
        Args:
            model_type: MiDaS model type ('DPT_Large', 'DPT_Hybrid', 'MiDaS_small')
            checkpoints_dir: Directory containing MiDaS checkpoint files
        """
        # Detect device with CUDA compatibility check
        self.device = self._get_device()
        self.checkpoints_dir = checkpoints_dir
        self.model_type = model_type
        
        print(f"Using device: {self.device}")
        
        # Create checkpoints directory if it doesn't exist
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            print(f"Created checkpoints directory: {checkpoints_dir}")
        
        # Load MiDaS model
        try:
            # Set torch hub directory to use checkpoints folder
            torch.hub.set_dir(checkpoints_dir)
            
            # Check if model already exists locally
            model_path = self._get_model_path(model_type)
            
            if model_path and os.path.exists(model_path):
                print(f"Loading MiDaS {model_type} from local checkpoint...")
                self.model = torch.hub.load(
                    "intel-isl/MiDaS", 
                    model_type, 
                    pretrained=True,
                    skip_validation=True,
                    force_reload=False
                )
            else:
                print(f"Downloading MiDaS {model_type} to {checkpoints_dir}...")
                print("This may take a few minutes on first run...")
                self.model = torch.hub.load(
                    "intel-isl/MiDaS", 
                    model_type, 
                    pretrained=True
                )
            
            # Move model to device with error handling
            try:
                self.model.to(self.device)
                self.model.eval()
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f" CUDA error detected: {e}")
                    print("Falling back to CPU...")
                    self.device = torch.device('cpu')
                    self.model.to(self.device)
                    self.model.eval()
                else:
                    raise
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            print(f" MiDaS {model_type} loaded successfully")
            print(f" Checkpoints stored in: {checkpoints_dir}")
            
        except Exception as e:
            print(f"Error loading MiDaS: {e}")
            print("Using fallback simple depth estimator")
            self.model = None
            self.transform = None
    
    def _get_device(self):
        """
        Get the best available device with CUDA compatibility check
        """
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        try:
            # Test CUDA availability
            device = torch.device('cuda')
            
            # Try a simple operation to check if CUDA actually works
            test_tensor = torch.zeros(1).to(device)
            _ = test_tensor + 1
            
            return device
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "no kernel image" in str(e):
                print(f" CUDA detected but not compatible: {e}")
                print("  Falling back to CPU mode")
                print("\n To fix CUDA issues:")
                print("   1. Check CUDA version: nvidia-smi")
                print("   2. Install matching PyTorch: https://pytorch.org/get-started/locally/")
                print("   3. Or set use_cpu=True in config\n")
                return torch.device('cpu')
            raise
    
    def _get_model_path(self, model_type):
        """Get the expected path for a model checkpoint"""
        # MiDaS models are stored in torch hub cache
        hub_dir = os.path.join(self.checkpoints_dir, 'intel-isl_MiDaS_master')
        
        if os.path.exists(hub_dir):
            return hub_dir
        
        return None
    
    def estimate(self, frame):
        """
        Estimate depth for a single frame
        
        Args:
            frame: RGB image (numpy array)
        
        Returns:
            Depth map as RGB image
        """
        if self.model is None:
            return self.fallback_depth(frame)
        
        try:
            # Prepare input
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth = prediction.cpu().numpy()
            
            # Normalize to 0-255
            depth_min = depth.min()
            depth_max = depth.max()
            
            if depth_max - depth_min > 0:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth)
            
            depth_normalized = (depth_normalized * 255).astype(np.uint8)
            
            # Apply colormap
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
            
            return depth_colored
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "kernel image" in str(e):
                print(f"  CUDA error during inference: {e}")
                print("  Switching to CPU mode...")
                
                # Switch to CPU
                self.device = torch.device('cpu')
                self.model.to(self.device)
                
                # Retry on CPU
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_batch = self.transform(img).to(self.device)
                    
                    with torch.no_grad():
                        prediction = self.model(input_batch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()
                    
                    depth = prediction.cpu().numpy()
                    depth_min = depth.min()
                    depth_max = depth.max()
                    
                    if depth_max - depth_min > 0:
                        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
                    else:
                        depth_normalized = np.zeros_like(depth)
                    
                    depth_normalized = (depth_normalized * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
                    
                    return depth_colored
                except Exception as cpu_error:
                    print(f" CPU inference also failed: {cpu_error}")
                    print("Using fallback depth estimator")
                    return self.fallback_depth(frame)
            else:
                raise
                
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return self.fallback_depth(frame)
    
    def fallback_depth(self, frame):
        """
        Simple fallback depth estimation using edge detection and gradients
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Compute gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        # Invert (closer objects have higher gradients)
        depth_map = 255 - gradient_magnitude
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        return depth_colored
    
    def estimate_with_overlay(self, frame, alpha=0.5):
        """
        Estimate depth and overlay on original frame
        
        Args:
            frame: RGB image
            alpha: Blending factor (0-1)
        
        Returns:
            Blended image
        """
        depth_colored = self.estimate(frame)
        blended = cv2.addWeighted(frame, 1 - alpha, depth_colored, alpha, 0)
        return blended
    
    def batch_estimate(self, frames):
        """
        Estimate depth for multiple frames
        
        Args:
            frames: List of RGB images
        
        Returns:
            List of depth maps
        """
        depth_maps = []
        for frame in frames:
            depth_map = self.estimate(frame)
            depth_maps.append(depth_map)
        return depth_maps


class UAVDepthEstimator:
    """
    Depth estimator optimized for UAV/aerial footage
    Provides better visualization for overhead/high-altitude views
    """
    
    def __init__(self):
        self.kernel_size = 5
        # Color mapping for altitude/height representation
        self.colormap = cv2.COLORMAP_JET  # Better for aerial views
    
    def estimate(self, frame):
        """
        Enhanced depth estimation for UAV footage
        Uses multiple techniques for better aerial depth perception
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge-based depth (structures, buildings, roads)
        edges = cv2.Canny(gray, 30, 100)
        edge_depth = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # 2. Texture-based depth (ground vs elevated features)
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(texture)
        texture_norm = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Brightness-based depth (shadows indicate elevation)
        # In aerial views, darker areas are often lower/in shadow
        brightness = gray.copy()
        
        # 4. Combine methods
        # Weight: 50% edges, 30% texture, 20% brightness
        edge_depth_norm = cv2.normalize(edge_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Combine with weighted average
        combined = (
            0.5 * edge_depth_norm.astype(np.float32) +
            0.3 * texture_norm.astype(np.float32) +
            0.2 * brightness.astype(np.float32)
        )
        
        # Normalize
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply bilateral filter to smooth while preserving edges
        combined = cv2.bilateralFilter(combined, 9, 75, 75)
        
        # Enhance contrast for better visibility
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        combined = clahe.apply(combined)
        
        # Apply colormap (JET: blue=far/low, red=close/high)
        depth_colored = cv2.applyColorMap(combined, self.colormap)
        
        # Add legend/scale indicator
        depth_colored = self._add_depth_legend(depth_colored)
        
        return depth_colored
    
    def _add_depth_legend(self, img):
        """Add a visual legend showing depth scale"""
        h, w = img.shape[:2]
        
        # Create legend bar (vertical, right side)
        legend_width = 30
        legend_height = 200
        legend_x = w - legend_width - 10
        legend_y = 10
        
        # Create gradient legend
        legend = np.zeros((legend_height, legend_width), dtype=np.uint8)
        for i in range(legend_height):
            legend[i, :] = int((legend_height - i) * 255 / legend_height)
        
        legend_colored = cv2.applyColorMap(legend, cv2.COLORMAP_JET)
        
        # Add legend to image
        img[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = legend_colored
        
        # Add labels
        cv2.putText(img, "High", (legend_x - 40, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, "Low", (legend_x - 35, legend_y + legend_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img


class LiteDepthEstimator:
    """
    Lightweight depth estimator for faster processing
    Uses simpler edge-based approach
    """
    
    def __init__(self):
        self.kernel_size = 5
    
    def estimate(self, frame):
        """Simple edge-based depth estimation"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Canny edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Distance transform from edges
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Normalize
        dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_normalized = dist_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_TURBO)
        
        return depth_colored


# Factory function
def create_depth_estimator(model_type="auto", lite=False, uav_mode=False, checkpoints_dir="checkpoints"):
    """
    Create depth estimator based on requirements
    
    Args:
        model_type: 'auto', 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'
        lite: Use lightweight estimator
        uav_mode: Use UAV/aerial optimized estimator
        checkpoints_dir: Directory to store/load model checkpoints
    
    Returns:
        DepthEstimator instance
    """
    if uav_mode:
        return UAVDepthEstimator()
    
    if lite:
        return LiteDepthEstimator()
    
    if model_type == "auto":
        # Check if CUDA is available
        if torch.cuda.is_available():
            model_type = "DPT_Hybrid"
        else:
            model_type = "MiDaS_small"
    
    return DepthEstimator(model_type, checkpoints_dir)