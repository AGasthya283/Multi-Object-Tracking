"""
CUDA Diagnostic and Fix Script
Automatically detects and suggests fixes for CUDA issues
"""

import sys
import subprocess

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    print("Checking NVIDIA driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(" NVIDIA driver is installed")
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"  System CUDA version: {cuda_version}")
                    return True, cuda_version
            return True, "Unknown"
        else:
            print(" NVIDIA driver not found")
            return False, None
    except FileNotFoundError:
        print(" nvidia-smi not found - NVIDIA driver not installed")
        return False, None

def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f" PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f" PyTorch CUDA is available")
            print(f"  PyTorch CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            
            # Get GPU info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            
            return True, torch.version.cuda
        else:
            print(" PyTorch CUDA is NOT available")
            print(f"  PyTorch version: {torch.__version__}")
            return False, None
            
    except ImportError:
        print(" PyTorch not installed")
        return False, None

def test_cuda_operation():
    """Test if CUDA actually works"""
    print("\nTesting CUDA operations...")
    try:
        import torch
        
        if not torch.cuda.is_available():
            print(" CUDA not available - skipping test")
            return False
        
        # Try a simple operation
        device = torch.device('cuda')
        x = torch.zeros(1).to(device)
        y = x + 1
        result = y.cpu().numpy()
        
        if result[0] == 1.0:
            print(" CUDA operation test PASSED")
            return True
        else:
            print(" CUDA operation test FAILED")
            return False
            
    except RuntimeError as e:
        print(f" CUDA operation test FAILED: {e}")
        if "no kernel image" in str(e):
            print("\n ERROR: CUDA kernel mismatch detected!")
            print("This means your PyTorch version doesn't match your GPU.")
        return False
    except Exception as e:
        print(f" Unexpected error: {e}")
        return False

def suggest_fix(driver_ok, driver_version, pytorch_ok, pytorch_cuda_version):
    """Suggest fixes based on diagnostic results"""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not driver_ok:
        print("\n NVIDIA Driver Issue")
        print("\nFix: Install NVIDIA drivers")
        print("  Ubuntu/Debian: sudo apt install nvidia-driver-535")
        print("  Windows: Download from https://www.nvidia.com/download/index.aspx")
        print("  After installation, reboot your system")
        return
    
    if not pytorch_ok:
        print("\n PyTorch CUDA Not Available")
        print("\nPossible causes:")
        print("  1. PyTorch installed without CUDA support")
        print("  2. CUDA version mismatch")
        
        if driver_version:
            major_version = driver_version.split('.')[0]
            if major_version == '12':
                print(f"\nFix: Install PyTorch with CUDA 12.1")
                print("  pip uninstall torch torchvision")
                print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            elif major_version == '11':
                print(f"\nFix: Install PyTorch with CUDA 11.8")
                print("  pip uninstall torch torchvision")
                print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("\nFix: Visit https://pytorch.org/get-started/locally/")
            print("  Select your system configuration and install PyTorch")
        return
    
    # Both driver and PyTorch are OK
    print("\n  System Configuration Looks Good")
    print(f"  NVIDIA Driver: {driver_version}")
    print(f"  PyTorch CUDA: {pytorch_cuda_version}")
    
    # Check version compatibility
    if driver_version and pytorch_cuda_version:
        driver_major = driver_version.split('.')[0]
        pytorch_major = pytorch_cuda_version.split('.')[0]
        
        if driver_major != pytorch_major:
            print(f"\n  Version Mismatch Warning")
            print(f"  Driver CUDA: {driver_version}")
            print(f"  PyTorch CUDA: {pytorch_cuda_version}")
            print("\nRecommendation: Install matching PyTorch version")
            if driver_major == '12':
                print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            elif driver_major == '11':
                print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

def provide_workarounds():
    """Provide immediate workarounds"""
    print("\n" + "="*60)
    print("IMMEDIATE WORKAROUNDS")
    print("="*60)
    
    print("\n1. Force CPU Mode (slowest but always works)")
    print("   Edit config.ini:")
    print("   [performance]")
    print("   force_cpu = true")
    
    print("\n2. Set Environment Variable")
    print("   Linux/Mac:")
    print("     CUDA_VISIBLE_DEVICES=\"\" streamlit run app.py")
    print("   Windows PowerShell:")
    print("     $env:CUDA_VISIBLE_DEVICES=\"\"; streamlit run app.py")
    
    print("\n3. Use Lighter Models")
    print("   Edit config.ini:")
    print("   [detection]")
    print("   default_model = yolov8n.pt")
    print("   [depth]")
    print("   use_lite = true")

def main():
    print("="*60)
    print("CUDA DIAGNOSTIC TOOL")
    print("="*60)
    print("\nThis tool will check your CUDA setup and suggest fixes.")
    print()
    
    # Run diagnostics
    driver_ok, driver_version = check_nvidia_driver()
    pytorch_ok, pytorch_cuda = check_pytorch()
    
    if pytorch_ok:
        cuda_works = test_cuda_operation()
    else:
        cuda_works = False
    
    # Provide recommendations
    suggest_fix(driver_ok, driver_version, pytorch_ok, pytorch_cuda)
    
    # If CUDA doesn't work, provide workarounds
    if not cuda_works:
        provide_workarounds()
    
    print("\n" + "="*60)
    print("For detailed help, see CUDA_TROUBLESHOOTING.md")
    print("="*60)
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled by user")
    except Exception as e:
        print(f"\nError running diagnostic: {e}")
