import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import mne
from scipy.signal import butter, filtfilt, hilbert
import cv2
import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional
import pywt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationState:
    """Holds the current state of visualization"""
    current_time: float = 0.0
    is_playing: bool = False
    seek_requested: bool = False
    seek_time: float = 0.0
    playback_speed: float = 1.0

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

            assert self.state.shape == (self.size, self.size), f"State shape mismatch: {self.state.shape}"
            return self.state
        except Exception as e:
            logger.error(f"Error in FrequencyBandProcessor.update: {e}")
            raise




class WaveletProcessor:
    """Handles Wavelet Decomposition and Reconstruction for EEG signals"""
    def __init__(self, wavelet='db4', levels=5):
        """
        Initialize the WaveletProcessor with the specified wavelet type and number of levels.

        :param wavelet: The wavelet type (e.g., 'db4' for Daubechies 4).
        :param levels: Number of decomposition levels.
        """
        self.wavelet = wavelet
        self.levels = levels

    def decompose(self, signal):
        """Decomposes a signal using wavelet transform.
        
        :param signal: 1D array of the signal to decompose.
        :return: List of wavelet coefficients.
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels)
        return coeffs

    def reconstruct(self, coeffs):
        """Reconstructs a signal from wavelet coefficients.
        
        :param coeffs: List of wavelet coefficients.
        :return: Reconstructed signal as a 1D array.
        """
        reconstructed_signal = pywt.waverec(coeffs, self.wavelet)
        return reconstructed_signal

    def filter_coeffs(self, coeffs, threshold=0.2):
        """Applies a threshold filter to wavelet coefficients.
        
        :param coeffs: List of wavelet coefficients.
        :param threshold: Threshold for soft filtering (fraction of max coefficient).
        :return: Filtered wavelet coefficients.
        """
        filtered_coeffs = []
        for coeff in coeffs:
            filtered_coeffs.append(pywt.threshold(coeff, threshold * np.max(coeff), mode='soft'))
        return filtered_coeffs

class ImageBacktracker:
    """Allows for input images to be processed and backtracked"""
    def __init__(self, processors):
        """
        Initialize the ImageBacktracker.

        :param processors: A dictionary of FrequencyBandProcessor instances (e.g., theta, alpha, beta, gamma).
        """
        self.processors = processors


    def process_image(self, image, amplitude=1.0, dt=0.033):
        try:
            # Ensure the image is grayscale and 2D
            logger.debug(f"Original image shape: {image.shape}")
            if len(image.shape) == 3:  # Convert RGB to grayscale if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) != 2:
                raise ValueError("Input image must be a 2D grayscale array.")

            logger.debug(f"Grayscale image shape: {image.shape}")

            # Resize the image to match the processor resolution
            resolution = self.processors['theta'].size  # Assume all processors use the same resolution
            input_image = cv2.resize(image, (resolution, resolution))
            logger.debug(f"Resized image shape: {input_image.shape}")

            # Normalize the image to [0, 1]
            img_min, img_max = input_image.min(), input_image.max()
            if img_max > img_min:
                input_image = (input_image - img_min) / (img_max - img_min)
            else:
                input_image = np.zeros_like(input_image)  # Handle uniform images
            logger.debug(f"Normalized image min: {input_image.min()}, max: {input_image.max()}")

            # Collect 2D states from processors
            processed_states = []
            for band, processor in self.processors.items():
                state = processor.update(amplitude, dt)  # This returns a 2D array
                if state.shape != (resolution, resolution):
                    raise ValueError(f"Processor {band} output shape mismatch: {state.shape}")
                logger.debug(f"Processor {band} output state shape: {state.shape}")
                processed_states.append(state)

            # Combine processed states into a single 2D array
            combined_state = np.sum(processed_states, axis=0)
            logger.debug(f"Combined state shape: {combined_state.shape}")

            # Backtrack: Convert the combined state into an interpretable image
            backtracked_image = self.backtrack_image(combined_state)
            logger.debug(f"Backtracked image shape: {backtracked_image.shape}")
            return backtracked_image

        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            raise


    def backtrack_image(self, combined_state):
        try:
            logger.debug(f"Input to backtrack_image shape: {combined_state.shape}")
            # Normalize the state to [0, 255]
            normalized_state = (combined_state - combined_state.min()) / (combined_state.max() - combined_state.min() + 1e-8)
            backtracked_image = (normalized_state * 255).astype(np.uint8)
            return backtracked_image
        except Exception as e:
            logger.error(f"Error in backtrack_image: {e}")
            raise


class EEGProcessor:
    """Handles EEG data processing"""
    def __init__(self):
        self.freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        self.window_size = 1.0  # 1 second window - increased for stable filtering
        self.min_samples = 100  # Minimum samples needed for processing
        
        # Initialize WaveletProcessor
        self.wavelet_processor = WaveletProcessor()

    def load_file(self, filepath: str) -> bool:
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            return True
        except Exception as e:
            logger.error(f"Failed to load EEG file: {e}")
            return False

    def get_data(self, channel: int, start_time: float) -> Optional[np.ndarray]:
        if self.raw is None:
            return None

        try:
            # Calculate sample points
            start_sample = int(start_time * self.sfreq)
            samples_needed = max(int(self.window_size * self.sfreq), self.min_samples)
            end_sample = start_sample + samples_needed

            # Handle end of file
            if end_sample >= self.raw.n_times:
                start_sample = 0
                end_sample = samples_needed

            # Get data
            data, _ = self.raw[channel, start_sample:end_sample]
            return data[0]

        except Exception as e:
            logger.error(f"Error getting EEG data: {e}")
            return None

    def process_bands(self, data: np.ndarray) -> Dict[str, float]:
        if data is None or len(data) < self.min_samples:
            return {band: 0.0 for band in self.freq_bands}

        try:
            # Normalize data
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)

            results = {}
            for band_name, (low, high) in self.freq_bands.items():
                # Bandpass filter
                nyq = self.sfreq / 2
                b, a = butter(4, [low / nyq, high / nyq], btype='band')
                filtered = filtfilt(b, a, data)

                # Get amplitude envelope
                analytic = hilbert(filtered)
                amplitude = np.abs(analytic)
                results[band_name] = np.mean(amplitude)

            return results
        except Exception as e:
            logger.error(f"Error processing frequency bands: {e}")
            return {band: 0.0 for band in self.freq_bands}

    def process_with_wavelet(self, data: np.ndarray) -> np.ndarray:
        """
        Process the EEG data with wavelet decomposition, filtering, and reconstruction.

        :param data: 1D array of EEG signal data.
        :return: Reconstructed signal after wavelet processing.
        """
        if data is None or len(data) < self.min_samples:
            return np.zeros_like(data)

        try:
            # Normalize data
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)

            # Decompose, filter, and reconstruct using the WaveletProcessor
            coeffs = self.wavelet_processor.decompose(data)
            filtered_coeffs = self.wavelet_processor.filter_coeffs(coeffs)
            reconstructed_data = self.wavelet_processor.reconstruct(filtered_coeffs)
            return reconstructed_data
        except Exception as e:
            logger.error(f"Error in wavelet processing: {e}")
            return np.zeros_like(data)

class Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Visualizer")
        
        # Initialize components
        self.eeg = EEGProcessor()
        self.state = VisualizationState()
        self.last_update = time.time()
        
        # Setup frequency band processors
        base_size = 32
        self.processors = {
            'theta': FrequencyBandProcessor(base_size, (4, 8)),
            'alpha': FrequencyBandProcessor(base_size, (8, 13)),
            'beta': FrequencyBandProcessor(base_size, (13, 30)),
            'gamma': FrequencyBandProcessor(base_size, (30, 100))
        }

        
        # Initialize the ImageBacktracker (after self.processors is defined)
        self.image_backtracker = ImageBacktracker(self.processors)
        
        # Set up the GUI
        self.setup_gui()

    def process_input_image(self, image_path):
        """
        Load an input image, process it through the EEG visualization, and display the backtracked result.
        """
        try:
            # Load the input image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return

            original_shape = image.shape  # Save original shape

            # Process the image through the backtracking pipeline
            backtracked_image = self.image_backtracker.process_image(image)

            # Resize backtracked image to the original size
            backtracked_image = cv2.resize(backtracked_image, (original_shape[1], original_shape[0]))

            # Apply color map for visualization
            colorized = cv2.applyColorMap(backtracked_image, cv2.COLORMAP_JET)

            # Convert to PhotoImage for display
            final_image = Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=final_image)

            # Update canvas
            self.canvas.delete("all")
            self.canvas.config(width=original_shape[1], height=original_shape[0])
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep reference
        except Exception as e:
            logger.error(f"Error processing input image: {e}")
            messagebox.showerror("Error", "Failed to process input image.")


    def load_and_process_image(self):
        """
        Opens a file dialog to load an input image and processes it.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.process_input_image(file_path)


    
    def setup_gui(self):
        # Main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Save image

        ttk.Button(control_frame, text="Save Image", command=self.save_image).pack(fill=tk.X, pady=5)



        # Controls
        ttk.Button(control_frame, text="Load EEG", command=self.load_eeg).pack(fill=tk.X, pady=5)
        
        # Channel selection
        ttk.Label(control_frame, text="Channel:").pack()
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var)
        self.channel_combo.pack(fill=tk.X, pady=5)
        
        # Playback controls
        controls = ttk.Frame(control_frame)
        controls.pack(fill=tk.X, pady=10)
        
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Process Input Image", command=self.load_and_process_image).pack(fill=tk.X, pady=5)

        # Time slider
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(
            control_frame, from_=0, to=100,
            variable=self.time_var,
            command=self.seek
        )
        self.time_slider.pack(fill=tk.X, pady=5)
        
        # Speed control
        ttk.Label(control_frame, text="Playback Speed:").pack()
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(
            control_frame, from_=0.1, to=3.0,
            variable=self.speed_var,
            command=lambda x: setattr(self.state, 'playback_speed', self.speed_var.get())
        )
        speed_slider.pack(fill=tk.X, pady=(0, 10))

        # Frequency band controls
        self.band_controls = {}
        for band in ['theta', 'alpha', 'beta', 'gamma']:
            frame = ttk.LabelFrame(control_frame, text=f"{band.title()} Band Controls")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            controls = {}
            
            # Wave Speed
            ttk.Label(frame, text="Wave Speed:").pack()
            speed_var = tk.DoubleVar(value=1.0)
            speed_slider = ttk.Scale(
                frame, from_=0.1, to=5.0,
                variable=speed_var,
                command=lambda v, b=band: self.update_band_param(b, 'wave_speed', float(v))
            )
            speed_slider.pack(fill=tk.X)
            controls['wave_speed'] = speed_var
            
            # Decay Rate
            ttk.Label(frame, text="Decay Rate:").pack()
            decay_var = tk.DoubleVar(value=0.1)
            decay_slider = ttk.Scale(
                frame, from_=0.0, to=1.0,
                variable=decay_var,
                command=lambda v, b=band: self.update_band_param(b, 'decay_rate', float(v))
            )
            decay_slider.pack(fill=tk.X)
            controls['decay_rate'] = decay_var
            
            # Coupling Strength
            ttk.Label(frame, text="Coupling Strength:").pack()
            coupling_var = tk.DoubleVar(value=0.3)
            coupling_slider = ttk.Scale(
                frame, from_=0.0, to=1.0,
                variable=coupling_var,
                command=lambda v, b=band: self.update_band_param(b, 'coupling_strength', float(v))
            )
            coupling_slider.pack(fill=tk.X)
            controls['coupling_strength'] = coupling_var
            
            self.band_controls[band] = controls
        
        # Canvas for visualization
        self.canvas = tk.Canvas(viz_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def save_image(self):
        """
        Saves the currently displayed image on the canvas.
        """
        try:
            # Get the current image from the canvas
            if hasattr(self.canvas, 'image'):
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
                )
                if save_path:
                    self.canvas.image._PhotoImage__photo.write(save_path)
                    messagebox.showinfo("Success", f"Image saved successfully: {save_path}")
            else:
                messagebox.showerror("Error", "No image available to save.")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            messagebox.showerror("Error", "Failed to save the image.")


    def load_eeg(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if filepath and self.eeg.load_file(filepath):
            self.channel_combo['values'] = self.eeg.raw.ch_names
            self.channel_combo.set(self.eeg.raw.ch_names[0])
            self.time_slider.configure(to=self.eeg.duration)
            messagebox.showinfo("Success", "EEG file loaded successfully")
        else:
            messagebox.showerror("Error", "Failed to load EEG file")
    
    def update_band_param(self, band: str, param: str, value: float):
        """Update a parameter for a specific frequency band"""
        if band in self.processors:
            setattr(self.processors[band], param, value)
    
    def toggle_playback(self):
        self.state.is_playing = not self.state.is_playing
        self.play_btn.configure(text="Pause" if self.state.is_playing else "Play")
        
        if self.state.is_playing:
            self.update()
    
    def seek(self, value):
        if not self.state.is_playing:
            self.state.seek_requested = True
            self.state.seek_time = float(value)
            self.state.current_time = float(value)
            self.update_visualization()
    
    def update(self):
        if not self.state.is_playing:
            return
        
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Update time
        self.state.current_time += dt * self.state.playback_speed
        if self.state.current_time >= self.eeg.duration:
            self.state.current_time = 0
        
        # Update slider
        self.time_var.set(self.state.current_time)
        
        # Update visualization
        self.update_visualization()
        
        # Schedule next update
        self.root.after(33, self.update)  # ~30 FPS
    
    def update_visualization(self):
        """Updates the visualization using wavelet-processed EEG data."""
        if not self.eeg.raw or not self.channel_var.get():
            return

        # Get current channel data
        channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
        data = self.eeg.get_data(channel_idx, self.state.current_time)
        if data is None:
            return

        # Apply wavelet processing to the signal
        wavelet_processed_data = self.eeg.process_with_wavelet(data)

        # Process frequency bands (after wavelet transformation)
        band_amplitudes = self.eeg.process_bands(wavelet_processed_data)

        # Update each frequency band for visualization
        resolution = 512
        final_image = np.zeros((resolution, resolution))

        # Enhanced weights and composition
        weights = {
            'theta': 0.15,  # Slower waves
            'alpha': 0.35,  # Dominant visualization
            'beta': 0.25,   # Medium frequency detail
            'gamma': 0.25   # Fast detail
        }
        dt = 0.033  # ~30 FPS

        # Process each band and compose the final image
        band_states = {}
        for band, weight in weights.items():
            processor = self.processors[band]
            state = processor.update(band_amplitudes[band], dt)
            resized = cv2.resize(state, (resolution, resolution))
            band_states[band] = resized
            final_image += weight * resized

        # Add cross-frequency coupling effects
        alpha_beta_coupling = np.multiply(band_states['alpha'], band_states['beta']) * 0.1
        theta_alpha_coupling = np.multiply(band_states['theta'], band_states['alpha']) * 0.1
        final_image += alpha_beta_coupling + theta_alpha_coupling

        # Enhanced normalization and contrast
        final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min() + 1e-8)
        final_image = np.power(final_image, 0.85)  # Gamma correction for better contrast
        final_image = (final_image * 255).astype(np.uint8)

        # Advanced visual effects
        # Edge enhancement
        edge_enhanced = cv2.Laplacian(final_image, cv2.CV_64F).astype(np.uint8)
        final_image = cv2.addWeighted(final_image, 0.9, edge_enhanced, 0.1, 0)

        # Multi-scale blending
        blur1 = cv2.GaussianBlur(final_image, (3, 3), 0)
        blur2 = cv2.GaussianBlur(final_image, (7, 7), 0)
        final_image = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

        # Final colorization with enhanced contrast
        colored = cv2.applyColorMap(final_image, cv2.COLORMAP_JET)

        # Convert to PhotoImage and display
        image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(width=resolution, height=resolution)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo  # Keep reference


def main():
    root = tk.Tk()
    app = Visualizer(root)
    root.geometry("1200x1100")
    root.mainloop()

if __name__ == "__main__":
    main()
