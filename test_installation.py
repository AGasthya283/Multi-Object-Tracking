"""
Test script to verify Multi-Object Tracking installation
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    print("=" * 50)
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'streamlit': 'streamlit',
        'ultralytics': 'ultralytics',
        'yt_dlp': 'yt-dlp',
        'filterpy': 'filterpy',
        'PIL': 'pillow',
    }
    
    failed = []
    
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print(f" {package:20s} - OK")
        except ImportError as e:
            print(f" {package:20s} - FAILED")
            failed.append(package)
    
    print("=" * 50)
    
    if failed:
        print(f"\n {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("   pip install " + " ".join(failed))
        return False
    else:
        print("\n All packages imported successfully!")
        return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f" CUDA is available")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            
            # Test CUDA operation
            print("\n  Testing CUDA operation...")
            try:
                device = torch.device('cuda')
                x = torch.zeros(1).to(device)
                y = x + 1
                result = y.cpu().numpy()
                
                if result[0] == 1.0:
                    print("  CUDA operation test PASSED")
                else:
                    print("  CUDA operation test FAILED")
            except RuntimeError as e:
                print(f"  CUDA operation test FAILED: {e}")
                if "no kernel image" in str(e):
                    print("\n  CUDA KERNEL MISMATCH DETECTED!")
                    print("  Your PyTorch CUDA version doesn't match your GPU.")
                    print("  Run: python check_cuda.py for fixes")
        else:
            print(" CUDA is not available - using CPU")
            print("  Note: Processing will be slower without GPU")
    except Exception as e:
        print(f" Error checking CUDA: {e}")
    
    print("=" * 50)

def test_yolo():
    """Test YOLOv8 model loading"""
    print("\nTesting YOLOv8 model...")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')
        print(" YOLOv8 model loaded successfully")
        print(f"  - Model type: {model.model.__class__.__name__}")
        print(f"  - Number of classes: {len(model.names)}")
    except Exception as e:
        print(f" Error loading YOLOv8: {e}")
        print("  Note: Model will be downloaded on first run")
    
    print("=" * 50)

def test_trackers():
    """Test tracker imports"""
    print("\nTesting tracker modules...")
    print("=" * 50)
    
    try:
        from trackers.sort import SORTTracker
        print("SORT tracker imported")
    except Exception as e:
        print(f"SORT tracker failed: {e}")
    
    try:
        from trackers.deepsort import DeepSORTTracker
        print("DeepSORT tracker imported")
    except Exception as e:
        print(f"DeepSORT tracker failed: {e}")
    
    try:
        from trackers.bytetrack import ByteTracker
        print("BYTETrack tracker imported")
    except Exception as e:
        print(f"BYTETrack tracker failed: {e}")
    
    print("=" * 50)

def test_depth_estimator():
    """Test depth estimator"""
    print("\nTesting depth estimator...")
    print("=" * 50)
    
    try:
        from depth_estimator import DepthEstimator, LiteDepthEstimator
        print("Depth estimator modules imported")
        
        # Test lite estimator (no dependencies)
        estimator = LiteDepthEstimator()
        print("LiteDepthEstimator initialized")
        
        # Test MiDaS estimator
        try:
            estimator = DepthEstimator(model_type="MiDaS_small")
            print("MiDaS depth estimator initialized")
        except Exception as e:
            print(f"MiDaS depth estimator warning: {e}")
            print("  Note: Will use fallback depth estimation")
        
    except Exception as e:
        print(f"Depth estimator failed: {e}")
    
    print("=" * 50)

def test_opencv():
    """Test OpenCV video capabilities"""
    print("\nTesting OpenCV...")
    print("=" * 50)
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        # Check video codecs
        fourcc_codecs = ['mp4v', 'XVID', 'H264', 'avc1']
        available_codecs = []
        
        for codec in fourcc_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                available_codecs.append(codec)
            except:
                pass
        
        print(f"  - Available codecs: {', '.join(available_codecs)}")
        
    except Exception as e:
        print(f"OpenCV test failed: {e}")
    
    print("=" * 50)

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("MULTI-OBJECT TRACKING SYSTEM - INSTALLATION TEST")
    print("=" * 50 + "\n")
    
    # Run tests
    imports_ok = test_imports()
    test_cuda()
    
    if imports_ok:
        test_yolo()
        test_trackers()
        test_depth_estimator()
        test_opencv()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if imports_ok:
        print("\nSystem is ready!")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
        print("\nOr use the quick start script:")
        print("   ./start.sh (Linux/Mac)")
        print("   start.bat (Windows)")
    else:
        print("\nSome tests failed")
        print("Please install missing dependencies and try again")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_all_tests()