#!/usr/bin/env python3
"""
Automatic setup script for Road Accident Detection
Downloads datasets and trains the model on first run
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_step(step, message):
    """Print formatted step message"""
    print(f"\n{'='*80}")
    print(f"STEP {step}: {message}")
    print(f"{'='*80}\n")

def check_model_exists():
    """Check if model file already exists"""
    model_path = project_root / "dict_models" / "vgg16_baseline.pth"
    return model_path.exists()

def install_dependencies():
    """Install required Python packages"""
    print_step(1, "Installing dependencies")
    
    requirements = [
        "gdown",
        "kagglehub",
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "Pillow"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
    
    print("All dependencies installed!")

def download_kaggle_dataset():
    """Download dataset from Kaggle"""
    print_step(2, "Downloading Kaggle dataset")
    
    try:
        import kagglehub
        
        dataset_ref = "ckay16/accident-detection-from-cctv-footage"
        datasets_path = project_root / "datasets"
        datasets_path.mkdir(exist_ok=True)
        
        print(f"Downloading {dataset_ref}...")
        path = kagglehub.dataset_download(handle=dataset_ref)
        print(f"Downloaded to: {path}")
        
        # Create symlink if needed
        target = datasets_path / "1"
        if not target.exists():
            import shutil
            shutil.copytree(path, target)
            print(f"Copied to: {target}")
        
        return True
    except Exception as e:
        print(f"Warning: Could not download Kaggle dataset: {e}")
        print("   Please download manually or configure Kaggle API")
        return False

def download_cadp_dataset():
    """Download CADP dataset from Google Drive"""
    print_step(3, "Downloading CADP dataset")
    
    try:
        import gdown
        
        file_id = '1AcGl5nyFoTmqNklxk-xpy2rb7tZT9EwL'
        output_path = project_root / "datasets" / "CADP"
        output_path.mkdir(parents=True, exist_ok=True)
        
        zip_path = output_path / "cadp.zip"
        
        print(f"Downloading CADP dataset...")
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
        
        print(f"Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        print(f"CADP dataset ready at: {output_path}")
        return True
    except Exception as e:
        print(f"Warning: Could not download CADP dataset: {e}")
        return False

def train_model():
    """Train the VGG16 baseline model"""
    print_step(4, "Training VGG16 model")
    
    # Check if datasets exist
    dataset_path = project_root / "datasets" / "1" / "data"
    if not dataset_path.exists():
        print("Dataset not found. Please download datasets first.")
        return False
    
    print("Starting model training...")
    print("   This may take a while (10-30 minutes depending on your hardware)")
    print("   Press Ctrl+C to cancel\n")
    
    # Import and run training
    try:
        # Change to project root
        os.chdir(project_root)
        
        # Run training script
        subprocess.run([
            sys.executable, "-m", 
            "src.models.model_train.baseline_vgg"
        ], check=True)
        
        # Check if model was created
        model_path = project_root / "models" / "vgg16_baseline.pth"
        if model_path.exists():
            # Move to dict_models directory
            dict_models_path = project_root / "dict_models"
            dict_models_path.mkdir(exist_ok=True)
            
            target_path = dict_models_path / "vgg16_baseline.pth"
            import shutil
            shutil.move(str(model_path), str(target_path))
            
            print(f"\nModel trained and saved to: {target_path}")
            return True
        else:
            print("Model file not found after training")
            return False
            
    except KeyboardInterrupt:
        print("\nTraining cancelled by user")
        return False
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def main():
    """Main setup function"""
    print("\n" + "="*80)
    print("ROAD ACCIDENT DETECTION - AUTOMATIC SETUP")
    print("="*80)
    
    # Check if model already exists
    if check_model_exists():
        print("\nModel already exists!")
        print(f"   Location: {project_root / 'dict_models' / 'vgg16_baseline.pth'}")
        
        response = input("\n❓ Do you want to retrain? (y/N): ").strip().lower()
        if response != 'y':
            print("\nSetup complete! You can start the service now.")
            return
    
    try:
        # Step 1: Install dependencies
        install_dependencies()
        
        # Step 2: Download Kaggle dataset
        kaggle_ok = download_kaggle_dataset()
        
        # Step 3: Download CADP dataset (optional)
        cadp_ok = download_cadp_dataset()
        
        # Step 4: Train model (only if Kaggle dataset was downloaded)
        if kaggle_ok:
            train_ok = train_model()
            
            if train_ok:
                print("\n" + "="*80)
                print("SETUP COMPLETE!")
                print("="*80)
                print("\nAll done! You can now:")
                print("   1. Start the cloud service: docker-compose up -d")
                print("   2. Test with video: python src/cloud/client.py --camera 0")
                print("   3. View dashboard: http://localhost:3000")
            else:
                print("\nTraining failed. Please check the errors above.")
        else:
            print("\nCannot train without dataset.")
            print("   Please configure Kaggle API or download dataset manually.")
            print("   See: https://github.com/Kaggle/kaggle-api")
    
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

