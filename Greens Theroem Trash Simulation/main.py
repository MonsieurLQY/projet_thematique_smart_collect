# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:28:15 2025

@author: sydne
"""

import os
import matplotlib.pyplot as plt

from sensor_simulation import simulate_rotating_ultrasonic_sensor_realistic as simulate_sensor
from geometry_utils import compute_area_green, compute_volume, compute_fill_percentage
from visualization import create_final_plot, create_gif_animation


# User can change output directory
OUTPUT_DIR = r"output"

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    print("[INFO] Generating simulated sensor sweep...")
    angles, heights = simulate_sensor()

    area_cm2 = compute_area_green(angles, heights)
    avg_height_cm, volume_cm3 = compute_volume(angles, heights)
    fill_pct = compute_fill_percentage(volume_cm3)

    print("\n----- TRASH BIN ANALYSIS RESULTS -----")
    print(f"Simulated Trash Surface Area: {area_cm2:.2f} cm²")
    print(f"Average Trash Height:        {avg_height_cm:.2f} cm")
    print(f"Approximate Trash Volume:     {volume_cm3:.2f} cm³")
    print(f"Estimated Bin Fill:           {fill_pct:.1f}%")
    print("---------------------------------------\n")

    print("[INFO] Creating final plot...")
    create_final_plot(angles, heights, area_cm2, avg_height_cm, volume_cm3, fill_pct, OUTPUT_DIR)

    print("[INFO] Creating sweep animation GIF...")
    create_gif_animation(angles, heights, OUTPUT_DIR)

    print("\n[INFO] COMPLETE!")
    print(f"Final output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
