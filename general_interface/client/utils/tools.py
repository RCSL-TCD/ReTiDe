import os
import shutil

def clean_results():
    # remove all files in /results folder
    if os.path.exists('results'):
        shutil.rmtree('results')
    os.makedirs('results/images', exist_ok=True)
