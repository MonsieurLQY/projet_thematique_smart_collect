# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:33:53 2025

@author: sydne
"""

import numpy as np

def simulate_rotating_ultrasonic_sensor_realistic(
        num_points=180,
        bin_radius=25,
        bin_height=50,
        trash_mean_height=25,
        trash_variability=8,
        sensor_noise=0.8,
        drift_amplitude=2,
    ):
    """
    Simulates realistic HC-SR04 sensor readings on a rotating arm,
    including trash unevenness, noise, lumps, and sensor drift.
    """
    angles = np.linspace(0, 2*np.pi, num_points)

    base_surface = trash_mean_height + trash_variability * np.sin(2 * angles)

    lump1 = 4 * np.exp(-((angles - 1.2*np.pi) ** 2) / 0.1)
    lump2 = 3 * np.exp(-((angles - 0.4*np.pi) ** 2) / 0.05)
    lump3 = 2.5 * np.exp(-((angles - 1.7*np.pi) ** 2) / 0.08)
    trash_surface = base_surface + lump1 + lump2 + lump3

    drift = drift_amplitude * np.sin(0.5 * angles)
    noise = np.random.normal(0, sensor_noise, num_points)

    heights = trash_surface + drift + noise
    heights = np.clip(heights, 0, bin_height)

    return angles, heights
