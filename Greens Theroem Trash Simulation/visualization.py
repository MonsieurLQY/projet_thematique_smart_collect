# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:35:55 2025

@author: sydne
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_final_plot(angles, heights, area, avg_height, volume, fill_pct, outdir):
    x = heights * np.cos(angles)
    y = heights * np.sin(angles)

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, linewidth=2)
    plt.fill(x, y, alpha=0.3)
    plt.title("Trash Surface Profile (Polar Sweep)")

    plt.text(0.05, 0.95, 
             f"Area: {area:.2f} cm²\nAvg Height: {avg_height:.2f} cm\nVolume: {volume:.2f} cm³\nFill: {fill_pct:.1f}%",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('equal')

    # Save final plot
    filepath = os.path.join(outdir, "final_plot.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

    # Save raw data to CSV
    csv_path = os.path.join(outdir, "raw_data.csv")
    np.savetxt(csv_path, np.column_stack((angles, heights)), delimiter=',', 
               header='angle_rad,height_cm', comments='')


def create_gif_animation(angles, heights, outdir):
    x = heights * np.cos(angles)
    y = heights * np.sin(angles)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(min(x)*1.2, max(x)*1.2)
    ax.set_ylim(min(y)*1.2, max(y)*1.2)
    ax.set_title("Sensor Sweep Animation")
    ax.set_aspect('equal')

    line, = ax.plot([], [], 'b-', linewidth=2)
    dot, = ax.plot([], [], 'ro')

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        dot.set_data([x[frame]], [y[frame]])
        return line, dot

    anim = FuncAnimation(fig, update, frames=len(angles), interval=30)
    gif_path = os.path.join(outdir, "scan_animation.gif")
    anim.save(gif_path, writer="pillow")
    plt.close()
