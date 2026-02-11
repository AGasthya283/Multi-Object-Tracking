"""
Download YOLOv8 Models
This script downloads YOLOv8 models to the models directory
"""

import os
from ultralytics import YOLO

def download_models():
    """Download YOLOv8 models to models directory"""
    
    # Create models directory
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f" Created models directory: {models_dir}")
    
    # Available models
    models = {
        'yolov8n.pt': 'YOLOv8 Nano (6 MB) - Fastest',
        'yolov8s.pt': 'YOLOv8 Small (22 MB) - Balanced',
        'yolov8m.pt': 'YOLOv8 Medium (50 MB) - Good accuracy',
        'yolov8l.pt': 'YOLOv8 Large (84 MB) - High accuracy',
        'yolov8x.pt': 'YOLOv8 Extra Large (131 MB) - Best accuracy'
    }
    
    print("\n" + "="*60)
    print("YOLOv8 Model Downloader")
    print("="*60)
    print("\nAvailable models:")
    for i, (model, desc) in enumerate(models.items(), 1):
        print(f"{i}. {model:15s} - {desc}")
    
    print("\n" + "="*60)
    print("Select models to download:")
    print("  - Enter numbers separated by comma (e.g., 1,2,3)")
    print("  - Enter 'all' to download all models")
    print("  - Enter 'default' to download yolov8n.pt and yolov8s.pt")
    print("="*60)
    
    choice = input("\nYour choice: ").strip().lower()
    
    # Determine which models to download
    to_download = []
    
    if choice == 'all':
        to_download = list(models.keys())
    elif choice == 'default':
        to_download = ['yolov8n.pt', 'yolov8s.pt']
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            model_list = list(models.keys())
            to_download = [model_list[i-1] for i in indices if 1 <= i <= len(models)]
        except:
            print(" Invalid input. Downloading default models.")
            to_download = ['yolov8n.pt', 'yolov8s.pt']
    
    # Download models
    print("\n" + "="*60)
    print("Downloading models...")
    print("="*60 + "\n")
    
    for model_name in to_download:
        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path):
            print(f" {model_name} already exists - skipping")
        else:
            print(f" Downloading {model_name}...")
            try:
                # Download model
                model = YOLO(model_name)
                
                # Move to models directory
                if os.path.exists(model_name):
                    import shutil
                    shutil.move(model_name, model_path)
                
                size = os.path.getsize(model_path) / (1024 * 1024)
                print(f" {model_name} downloaded successfully ({size:.1f} MB)")
            except Exception as e:
                print(f" Error downloading {model_name}: {e}")
    
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    # List downloaded models
    if os.path.exists(models_dir):
        downloaded = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        if downloaded:
            print(f"\nFound {len(downloaded)} model(s) in 'models' directory:")
            for model in sorted(downloaded):
                size = os.path.getsize(os.path.join(models_dir, model)) / (1024 * 1024)
                print(f"  - {model:20s} ({size:.1f} MB)")
            print("\n Models ready to use!")
            print("\nTo start the application, run:")
            print("  streamlit run app.py")
        else:
            print("\n No models found")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        download_models()
    except KeyboardInterrupt:
        print("\n\n Download cancelled by user")
    except Exception as e:
        print(f"\n Error: {e}")