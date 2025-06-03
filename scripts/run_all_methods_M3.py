import os
import sys
import runpy

# Get current script directory
DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
print(DIR)
os.chdir(DIR)
sys.path.append(DIR)

# Add src to path (if you need to import modules from there)
if DIR not in sys.path:
    sys.path.append(DIR)
# Add src to path (if you need to import modules from there)
if os.path.join(DIR, "src") not in sys.path:
    sys.path.append(os.path.join(DIR, "src"))


dataset = "M3"

# Define path to scripts
scripts_dir = os.path.join(DIR, "scripts",dataset)

# Sort to ensure consistent execution order
script_files = sorted([
    f for f in os.listdir(scripts_dir)
    if f.endswith('.py') and not f.startswith('__')
])

# Run each script
print(script_files)
for script in script_files:
    script_path = os.path.join(scripts_dir, script)
    print(f"Running: {script_path}")
    runpy.run_path(script_path, run_name="__main__")