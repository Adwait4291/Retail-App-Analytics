# setup_ml_project.py
import os
import subprocess
import sys
from pathlib import Path

def create_file(path):
    """Create an empty file"""
    with open(path, 'w') as f:
        pass
    print(f"Created: {path}")

def create_ml_project_structure(project_name):
    """Create a complete ML project structure with all necessary files"""
    
    # Create project directory
    project_dir = Path(project_name)
    if project_dir.exists():
        print(f"Warning: Directory {project_name} already exists. Files may be overwritten.")
    
    # Define the directory structure
    directories = [
        "src",
        "src/data",
        "src/models",
        "src/api",
        "tests",
        "tests/data",
        "tests/models",
        "tests/api",
        "data",
        "models",
        "logs",
        "notebooks",
        "docs"
    ]
    
    # Create all directories
    for directory in directories:
        os.makedirs(project_dir / directory, exist_ok=True)
        print(f"Created directory: {project_dir / directory}")
    
    # Create empty __init__.py files in all Python package directories
    for directory in ["src", "src/data", "src/models", "src/api", 
                     "tests", "tests/data", "tests/models", "tests/api"]:
        create_file(project_dir / directory / "__init__.py")
    
    # Create empty Python modules
    py_files = [
        "src/data/preprocessing.py",
        "src/data/feature_engineering.py",
        "src/models/train.py",
        "src/models/predict.py",
        "src/api/app.py",
        "main.py",
        "api_server.py",
        "create_test_data.py"
    ]
    
    for py_file in py_files:
        create_file(project_dir / py_file)
    
    # Create configuration and documentation files
    config_files = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example"
    ]
    
    for config_file in config_files:
        create_file(project_dir / config_file)
    
    # Create a basic Jupyter notebook
    notebooks_dir = project_dir / "notebooks"
    create_file(notebooks_dir / "exploratory_analysis.ipynb")
    
    print(f"\nProject structure for '{project_name}' created successfully!")

def setup_environment(create_new_env, project_name):
    """Set up Python environment for the project"""
    
    if create_new_env:
        print(f"Creating new conda environment '{project_name}'...")
        try:
            subprocess.run(["conda", "create", "-n", project_name, "python=3.10", "-y"], check=True)
            print(f"Environment '{project_name}' created successfully.")
            print(f"To activate the environment, run: conda activate {project_name}")
            print("Then install the required packages: pip install -r requirements.txt")
        except subprocess.CalledProcessError as e:
            print(f"Error creating environment: {e}")
        except FileNotFoundError:
            print("Conda not found. Please install Conda or use the base environment.")
    else:
        print("Using base environment. To install dependencies:")
        print(f"cd {project_name}")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    # Ask for project name
    project_name = input("Enter your ML project name: ").strip()
    if not project_name:
        project_name = "ml-project"
        print(f"Using default project name: {project_name}")
    
    # Ask about environment
    create_new_env = input("Create a new conda environment? (y/n): ").lower().startswith('y')
    
    # Create project structure
    create_ml_project_structure(project_name)
    
    # Setup environment
    setup_environment(create_new_env, project_name)