# --------------------------------------------------------------------------- #
# ███╗   ███╗██╗   ██╗███████╗██╗████████╗██████╗  █████╗ ███╗   ██╗███████╗
# ████╗ ████║██║   ██║██╔════╝██║╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝
# ██╔████╔██║██║   ██║███████╗██║   ██║   ██████╔╝███████║██╔██╗ ██║███████╗
# ██║╚██╔╝██║██║   ██║╚════██║██║   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║
# ██║ ╚═╝ ██║╚██████╔╝███████║██║   ██║   ██║  ██║██║  ██║██║ ╚████║███████║
# ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝
#                                                                           
# MUSITRANS: a one hobbyists project for automatic music transcription
# author: Adam Kurowski
# e-mail: akkurowski@gmail.com
# date:   24.07.2021
# --------------------------------------------------------------------------- #

# Perform imports
from imports import *
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import pandas as pd

if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

# Locking the seed of the random number generator
random.seed(42)

# Display the header
print('---------------------------------------------------------')
print('MUSITRANS - a simple, heuristic music transcriber')
print('author: Adam Kurowski')
print('e-mail: akkurowski@gmail.com')
print('date:   20.07.2021')
print('---------------------------------------------------------')
print()

# --------------------------------------------------------------------------- #
# Settings readout:
# --------------------------------------------------------------------------- #

settings = obtain_settings_structure("setings/default.ini")

# Creation of all declared directories.
for dir_name in settings['DIRECTORIES']:
    if not os.path.isdir(settings['DIRECTORIES'][dir_name]):
        os.makedirs(settings['DIRECTORIES'][dir_name])

# --------------------------------------------------------------------------- #
# Main interaction loop
# --------------------------------------------------------------------------- #

# Interruptable command execution adapter.
def execute_command(command_func, args=[], kwargs={}):
    try:
        command_func(*args,**kwargs)
    except KeyboardInterrupt:
        print()
        print('command execution was aborted')
        print()

# Menu items for the user:
items = []
items.append('show the oversampled spectrogram')
items.append('generate prechromograms')
items.append('show prechromograms')
items.append('track melodies')
items.append('show melody tracking results')
items.append('synthesize tracking results')
items.append('end the script execution')

# The Main Interaction Loop:
while True:
    print()
    ans = ask_user_for_an_option_choice('Choose your action:', 'Action number:', items)
    print()
    
    if ans == 'show the oversampled spectrogram':
        execute_command(cmd_show_oversampled_spectrograms, [settings])
    
    elif ans == 'generate prechromograms':
        execute_command(cmd_generate_prechromograms, [settings])
    
    elif ans == 'show prechromograms':
        execute_command(cmd_show_prechromogram, [settings])
    
    elif ans == 'track melodies':
        execute_command(cmd_detect_f0, [settings])
    
    elif ans == 'show melody tracking results':
        execute_command(cmd_show_melody_tracking_results, [settings])
    
    elif ans == 'synthesize tracking results':
        execute_command(synthesize_tracking_results, [settings])
    
    elif ans == 'end the script execution':
        print()
        print('The script execution was terminated.')
        print()
        break
    
    else:
        raise RuntimeError('a bad option was picked')
    print()
