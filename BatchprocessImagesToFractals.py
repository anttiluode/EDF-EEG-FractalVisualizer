import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrequencyBandProcessor:
    """Handles frequency band processing and visualization with enhanced wave dynamics"""
    def __init__(self, size: int, freq_range: tuple):
        self.size = size
        self.freq_range = freq_range
        
        # State variables
        self.state = np.zeros((size, size))
        self.phase = np.random.random((size, size)) * 2 * np.pi
        self.momentum = np.zeros((size, size))
        
        # Wave parameters
        self.decay_rate = 0.1
        self.wave_speed = (freq_range[0] + freq_range[1]) / 2
        self.coupling_strength = 0.3
        
        # Create distance matrices for wave propagation
        x = np.linspace(-np.pi, np.pi, size)
        y = np.linspace(-np.pi, np.pi, size)
        X, Y = np.meshgrid(x, y)
        self.center_x, self.center_y = size // 2, size // 2
        self.distance = np.sqrt((X - X[self.center_x, self.center_y])**2 + 
                              (Y - Y[self.center_y, self.center_y])**2)
        
        # Create directional matrices for wave motion
        self.angle = np.arctan2(Y - Y[self.center_x, self.center_y],
                              X - X[self.center_x, self.center_y])
    
    def update(self, amplitude: float, dt: float) -> np.ndarray:
        try:
            # Natural decay
            self.state *= (1 - self.decay_rate * dt)

            # Update phase with frequency-dependent speed
            self.phase += dt * 2 * np.pi * self.wave_speed

            # Generate wave patterns
            radial_wave = np.sin(self.distance - self.phase) * np.exp(-self.distance * 0.15)
            spiral_wave = np.sin(self.distance + self.angle - self.phase) * np.exp(-self.distance * 0.1)

            # Combine waves with amplitude modulation
            wave = (radial_wave * 0.6 + spiral_wave * 0.4) * amplitude

            # Add momentum for smooth transitions
            self.momentum = 0.95 * self.momentum + 0.05 * (wave - self.state)
            self.state += self.momentum * dt * 3.0

            # Add subtle spatial filtering for smoothness
            kernel_size = max(3, self.size // 16)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.state = cv2.GaussianBlur(self.state, (kernel_size, kernel_size), 0)

            return self.state
        except Exception as e:
            logger.error(f"Error in FrequencyBandProcessor.update: {e}")
            raise

class ImageBacktracker:
    def __init__(self, processors):
        self.processors = processors

    def process_image(self, image, amplitude=1.0, dt=0.033):
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) != 2:
                raise ValueError("Input image must be a 2D grayscale array.")

            resolution = self.processors['theta'].size
            input_image = cv2.resize(image, (resolution, resolution))

            img_min, img_max = input_image.min(), input_image.max()
            if img_max > img_min:
                input_image = (input_image - img_min) / (img_max - img_min)
            else:
                input_image = np.zeros_like(input_image)

            processed_states = []
            for band, processor in self.processors.items():
                state = processor.update(amplitude, dt)
                processed_states.append(state)

            combined_state = np.sum(processed_states, axis=0)
            backtracked_image = self.backtrack_image(combined_state)
            return backtracked_image

        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            raise

    def backtrack_image(self, combined_state):
        try:
            normalized_state = (combined_state - combined_state.min()) / (combined_state.max() - combined_state.min() + 1e-8)
            backtracked_image = (normalized_state * 255).astype(np.uint8)
            return backtracked_image
        except Exception as e:
            logger.error(f"Error in backtrack_image: {e}")
            raise

class ModelMakerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractal Image Generator and Model Maker")
        
        # Initialize processors
        base_size = 32
        self.processors = {
            'theta': FrequencyBandProcessor(base_size, (4, 8)),
            'alpha': FrequencyBandProcessor(base_size, (8, 13)),
            'beta': FrequencyBandProcessor(base_size, (13, 30)),
            'gamma': FrequencyBandProcessor(base_size, (30, 100))
        }
        
        self.image_backtracker = ImageBacktracker(self.processors)
        self.setup_gui()
        
    def setup_gui(self):
        # Main frames
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Help section
        help_text = """
        This tool automates the process of:
        1. Loading images from 'input_images' folder
        2. Creating fractal versions using the same process as the visualizer
        3. Saving fractals to 'fractal_images' folder
        4. Preparing image pairs for training
        
        Steps to use:
        1. Place original images in 'input_images' folder
        2. Click 'Process Images' to create fractals
        """
        help_label = ttk.Label(self.main_frame, text=help_text, wraplength=400, justify=tk.LEFT)
        help_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Folder paths
        ttk.Label(self.main_frame, text="Input folder: input_images/").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(self.main_frame, text="Output folder: fractal_images/").grid(row=2, column=0, sticky=tk.W)
        
        # Process button
        ttk.Button(self.main_frame, text="Process Images", command=self.process_images).grid(row=3, column=0, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.main_frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=2)

    def process_images(self):
        try:
            # Ensure directories exist
            input_dir = "input_images"
            output_dir = "fractal_images"
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get list of images
            image_files = [f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_files:
                messagebox.showwarning("No Images", "No images found in input_images folder.")
                return
            
            # Process each image
            for i, filename in enumerate(image_files):
                try:
                    # Load image
                    input_path = os.path.join(input_dir, filename)
                    image = cv2.imread(input_path)
                    
                    if image is None:
                        logger.error(f"Failed to load image: {filename}")
                        continue
                    
                    # Process image
                    original_size = image.shape[:2]
                    fractal_image = self.image_backtracker.process_image(image)
                    
                    # Resize back to original dimensions
                    fractal_image = cv2.resize(fractal_image, (original_size[1], original_size[0]))
                    
                    # Apply color map
                    colored = cv2.applyColorMap(fractal_image, cv2.COLORMAP_JET)
                    
                    # Save processed image
                    output_path = os.path.join(output_dir, f"fractal_{filename}")
                    cv2.imwrite(output_path, colored)
                    
                    # Update progress
                    progress = (i + 1) / len(image_files) * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"Processing {i+1}/{len(image_files)}: {filename}")
                    self.root.update()
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    continue
            
            self.status_var.set("Processing complete!")
            messagebox.showinfo("Success", "All images processed successfully!")
            
        except Exception as e:
            logger.error(f"Error in process_images: {e}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = ModelMakerApp(root)
    root.geometry("600x400")
    root.mainloop()

if __name__ == "__main__":
    main()