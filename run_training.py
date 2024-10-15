###################################################
#
#   Script to launch the training
#
##################################################

import os
import sys
import configparser
import time
import subprocess  # Import subprocess for better command execution

start = time.time()

# Check for argument
if len(sys.argv) != 2:
    print("Usage: python3 run_training.py <configuration_file>")
    exit(1)

# config file to read from
config_name = sys.argv[1]
config = configparser.RawConfigParser()

# Read the configuration file
with open(config_name, 'r') as config_file:
    config.read_file(config_file)

# ===========================================
# Name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')  # std output on log file?

# Setting GPU flags for non-Windows platforms
run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

# Create a folder for the results
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
os.makedirs(result_dir, exist_ok=True)  # Create directory if it does not exist

# Copy the configuration file to the results folder
print("Copying the configuration file to the results folder")
if sys.platform == 'win32':
    subprocess.run(['copy', config_name, f'.\\{name_experiment}\\{name_experiment}_configuration.txt'], shell=True)
else:
    subprocess.run(['cp', config_name, f'./{name_experiment}/{name_experiment}_configuration.txt'])

# Run the experiment with the configuration file as an argument
if nohup:
    print("\n2. Run the training on GPU with nohup")
    subprocess.run(f'{run_GPU} nohup python3 -u ./src/retina_unet_training.py {config_name} > ./{name_experiment}/{name_experiment}_training.nohup', shell=True)
else:
    print("\n2. Run the training on GPU (no nohup)")
    subprocess.run(f'{run_GPU} python3 ./src/retina_unet_training.py {config_name}', shell=True)

print("Training script has been executed.")