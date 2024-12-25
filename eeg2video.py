import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import mne
from scipy.signal import butter, filtfilt, hilbert
import cv2
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# 1) FRACTAL EEG VISUALIZER CODE (EXACTLY AS PROVIDED)
###############################################################################

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
                                (Y - Y[self.center_x, self.center_y])**2)
        
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
        self.wavelet = wavelet
        self.levels = levels

    def decompose(self, signal):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels)
        return coeffs

    def reconstruct(self, coeffs):
        reconstructed_signal = pywt.waverec(coeffs, self.wavelet)
        return reconstructed_signal

    def filter_coeffs(self, coeffs, threshold=0.2):
        filtered_coeffs = []
        for coeff in coeffs:
            filtered_coeffs.append(pywt.threshold(coeff, threshold * np.max(coeff), mode='soft'))
        return filtered_coeffs

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
        self.window_size = 1.0
        self.min_samples = 100
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
            start_sample = int(start_time * self.sfreq)
            samples_needed = max(int(self.window_size * self.sfreq), self.min_samples)
            end_sample = start_sample + samples_needed

            if end_sample >= self.raw.n_times:
                start_sample = 0
                end_sample = samples_needed

            data, _ = self.raw[channel, start_sample:end_sample]
            return data[0]

        except Exception as e:
            logger.error(f"Error getting EEG data: {e}")
            return None

    def process_bands(self, data: np.ndarray) -> Dict[str, float]:
        if data is None or len(data) < self.min_samples:
            return {band: 0.0 for band in self.freq_bands}
        try:
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            results = {}
            for band_name, (low, high) in self.freq_bands.items():
                nyq = self.sfreq / 2
                b, a = butter(4, [low / nyq, high / nyq], btype='band')
                filtered = filtfilt(b, a, data)

                analytic = hilbert(filtered)
                amplitude = np.abs(analytic)
                results[band_name] = np.mean(amplitude)

            return results
        except Exception as e:
            logger.error(f"Error processing frequency bands: {e}")
            return {band: 0.0 for band in self.freq_bands}

    def process_with_wavelet(self, data: np.ndarray) -> np.ndarray:
        if data is None or len(data) < self.min_samples:
            return np.zeros_like(data)
        try:
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            coeffs = self.wavelet_processor.decompose(data)
            filtered_coeffs = self.wavelet_processor.filter_coeffs(coeffs)
            reconstructed_data = self.wavelet_processor.reconstruct(filtered_coeffs)
            return reconstructed_data
        except Exception as e:
            logger.error(f"Error in wavelet processing: {e}")
            return np.zeros_like(data)

class ImageBacktracker:
    """Allows for input images to be processed and backtracked"""
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
                if state.shape != (resolution, resolution):
                    raise ValueError(f"Processor {band} output shape mismatch: {state.shape}")
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

###############################################################################
# 2) FRACTAL DECODER (EnhancedUNet, EXACT as Provided)
###############################################################################

class EnhancedUNet(nn.Module):
    def __init__(self):
        super(EnhancedUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._double_conv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, 1)
        
        # Frequency analysis branch
        self.freq_conv1 = self._double_conv(1, 32)
        self.freq_conv2 = self._double_conv(32, 64)
        self.freq_pool = nn.MaxPool2d(2)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        # Frequency analysis path
        f1 = self.freq_conv1(x)
        f2 = self.freq_conv2(self.freq_pool(f1))
        
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        
        s1 = F.interpolate(out, scale_factor=0.25)
        s2 = F.interpolate(out, scale_factor=0.5)
        
        return out, [s1, s2, out]

###############################################################################
# 3) VISUALIZER GUI (PRIMARY WINDOW)
#    We'll open a second window for Model/Video Controls, and a third window for extra canvas
###############################################################################

class Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Visualizer")

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
        self.image_backtracker = ImageBacktracker(self.processors)

        # Decoder model references
        self.decoder_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For EXACT decoding logic, use the same transform from the fractal decoder
        self.decode_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.setup_gui()

    def setup_gui(self):
        ##########################
        # LEFT CONTROL PANEL
        ##########################
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ##########################
        # RIGHT VISUALIZATION FRAME
        ##########################
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Save/Load EEG Buttons
        ttk.Button(control_frame, text="Save Image", command=self.save_image).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Load EEG", command=self.load_eeg).pack(fill=tk.X, pady=5)
        
        # Channel Selection
        ttk.Label(control_frame, text="Channel:").pack()
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var)
        self.channel_combo.pack(fill=tk.X, pady=5)
        
        # Playback Controls
        controls = ttk.Frame(control_frame)
        controls.pack(fill=tk.X, pady=10)
        
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Input Image -> fractal
        ttk.Button(control_frame, text="Process Input Image", command=self.load_and_process_image).pack(fill=tk.X, pady=5)

        # Time Slider
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(
            control_frame, from_=0, to=100,
            variable=self.time_var,
            command=self.seek
        )
        self.time_slider.pack(fill=tk.X, pady=5)
        
        # Playback Speed
        ttk.Label(control_frame, text="Playback Speed:").pack()
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(
            control_frame, from_=0.1, to=3.0,
            variable=self.speed_var,
            command=lambda x: setattr(self.state, 'playback_speed', self.speed_var.get())
        )
        speed_slider.pack(fill=tk.X, pady=(0, 10))

        # Frequency Band Controls
        self.band_controls = {}
        for band in ['theta', 'alpha', 'beta', 'gamma']:
            frame = ttk.LabelFrame(control_frame, text=f"{band.title()} Band Controls")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            controls = {}
            ttk.Label(frame, text="Wave Speed:").pack()
            speed_var = tk.DoubleVar(value=1.0)
            speed_slider = ttk.Scale(
                frame, from_=0.1, to=5.0,
                variable=speed_var,
                command=lambda v, b=band: self.update_band_param(b, 'wave_speed', float(v))
            )
            speed_slider.pack(fill=tk.X)
            controls['wave_speed'] = speed_var
            
            ttk.Label(frame, text="Decay Rate:").pack()
            decay_var = tk.DoubleVar(value=0.1)
            decay_slider = ttk.Scale(
                frame, from_=0.0, to=1.0,
                variable=decay_var,
                command=lambda v, b=band: self.update_band_param(b, 'decay_rate', float(v))
            )
            decay_slider.pack(fill=tk.X)
            controls['decay_rate'] = decay_var
            
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

        # Two new buttons to open second/third windows
        ttk.Button(control_frame, text="Open Model/Video Window", command=self.open_model_video_window).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Open Extra Canvas Window", command=self.open_extra_canvas_window).pack(fill=tk.X, pady=10)

        ##########################
        # CANVAS FOR VISUALIZATION
        ##########################
        self.canvas = tk.Canvas(viz_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    ################################################################
    # OPEN SECOND (MODEL/VIDEO) WINDOW
    ################################################################
    def open_model_video_window(self):
        """
        Creates a second Toplevel window that holds the
        model/video control buttons (load model, save video, etc.).
        """
        model_win = tk.Toplevel(self.root)
        model_win.title("Fractal Decoder / Video Controls")
        model_win.geometry("400x300")

        ttk.Button(model_win, text="Load Decoder Model", command=self.load_decoder_model).pack(fill=tk.X, pady=5)
        ttk.Button(model_win, text="Save Video With Model", command=self.save_video_with_model).pack(fill=tk.X, pady=5)
        ttk.Button(model_win, text="Play Video With Model", command=self.play_video_with_model).pack(fill=tk.X, pady=5)

    ################################################################
    # OPEN THIRD (EXTRA CANVAS) WINDOW
    ################################################################
    def open_extra_canvas_window(self):
        """
        Creates a third Toplevel window with a blank canvas for anything extra.
        """
        canvas_win = tk.Toplevel(self.root)
        canvas_win.title("Extra Canvas Window")
        canvas_win.geometry("600x600")

        # an extra canvas
        self.extra_canvas = tk.Canvas(canvas_win, width=512, height=512, bg='white')
        self.extra_canvas.pack(padx=10, pady=10)

        # You can draw or place images on self.extra_canvas in the future.

    ################################################################
    # EEG LOAD/SAVE/PROCESS
    ################################################################
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

    def save_image(self):
        try:
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

    def load_and_process_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.process_input_image(file_path)

    def process_input_image(self, image_path):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return

            original_shape = image.shape
            amplitude = 1.0
            dt = 0.033
            backtracked_image = self.image_backtracker.process_image(image, amplitude=amplitude, dt=dt)
            backtracked_image = cv2.resize(backtracked_image, (original_shape[1], original_shape[0]))
            colorized = cv2.applyColorMap(backtracked_image, cv2.COLORMAP_JET)

            final_image = Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=final_image)

            self.canvas.delete("all")
            self.canvas.config(width=original_shape[1], height=original_shape[0])
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo
        except Exception as e:
            logger.error(f"Error processing input image: {e}")
            messagebox.showerror("Error", "Failed to process input image.")

    def update_band_param(self, band: str, param: str, value: float):
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
        
        self.state.current_time += dt * self.state.playback_speed
        if self.state.current_time >= self.eeg.duration:
            self.state.current_time = 0
        
        self.time_var.set(self.state.current_time)
        
        self.update_visualization()
        self.root.after(33, self.update)  # ~30 FPS

    def update_visualization(self):
        if not self.eeg.raw or not self.channel_var.get():
            return

        channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
        data = self.eeg.get_data(channel_idx, self.state.current_time)
        if data is None:
            return

        wavelet_processed_data = self.eeg.process_with_wavelet(data)
        band_amplitudes = self.eeg.process_bands(wavelet_processed_data)

        resolution = 512
        final_image = np.zeros((resolution, resolution))

        weights = {
            'theta': 0.15,
            'alpha': 0.35,
            'beta': 0.25,
            'gamma': 0.25
        }
        dt = 0.033

        band_states = {}
        for band, weight in weights.items():
            processor = self.processors[band]
            state = processor.update(band_amplitudes[band], dt)
            resized = cv2.resize(state, (resolution, resolution))
            band_states[band] = resized
            final_image += weight * resized

        # coupling
        alpha_beta_coupling = np.multiply(band_states['alpha'], band_states['beta']) * 0.1
        theta_alpha_coupling = np.multiply(band_states['theta'], band_states['alpha']) * 0.1
        final_image += alpha_beta_coupling + theta_alpha_coupling

        # normalize
        final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min() + 1e-8)
        final_image = np.power(final_image, 0.85)
        final_image = (final_image * 255).astype(np.uint8)

        # edge enhance
        edge_enhanced = cv2.Laplacian(final_image, cv2.CV_64F).astype(np.uint8)
        final_image = cv2.addWeighted(final_image, 0.9, edge_enhanced, 0.1, 0)

        # multi-scale blending
        blur1 = cv2.GaussianBlur(final_image, (3, 3), 0)
        blur2 = cv2.GaussianBlur(final_image, (7, 7), 0)
        final_image = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

        colored = cv2.applyColorMap(final_image, cv2.COLORMAP_JET)

        image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.delete("all")
        self.canvas.config(width=resolution, height=resolution)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo

    ################################################################
    # METHODS CALLED BY MODEL/VIDEO WINDOW
    ################################################################
    def load_decoder_model(self):
        """Load the fractal decoder PyTorch .pth file and store in self.decoder_model."""
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.decoder_model = EnhancedUNet().to(self.device)
            checkpoint = torch.load(file_path, map_location=self.device)
            self.decoder_model.load_state_dict(checkpoint['model_state_dict'])
            self.decoder_model.eval()
            messagebox.showinfo("Success", f"Decoder model loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading decoder model: {e}")
            messagebox.showerror("Error", "Failed to load decoder model")

    def save_video_with_model(self):
        """Generate a fractal-decoded video using the loaded model, saving to a file."""
        if not self.eeg.raw:
            messagebox.showerror("Error", "No EEG loaded.")
            return
        if not self.decoder_model:
            messagebox.showerror("Error", "No decoder model loaded.")
            return
        
        out_file = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not out_file:
            return
        
        try:
            start_time = float(simpledialog.askstring("Input", "Start time (seconds)?", initialvalue="0.0"))
            end_time = float(simpledialog.askstring("Input", "End time (seconds)?", initialvalue="10.0"))
            fps = float(simpledialog.askstring("Input", "FPS?", initialvalue="30"))
        except:
            messagebox.showerror("Error", "Invalid input for time/fps.")
            return

        try:
            channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
        except:
            channel_idx = 0

        out_width, out_height = 256, 256
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_file, fourcc, fps, (out_width, out_height), isColor=False)

        total_frames = int((end_time - start_time) * fps)
        dt = 1.0 / fps
        current_time = start_time

        band_weights = {'theta': 0.15, 'alpha': 0.35, 'beta': 0.25, 'gamma': 0.25}
        freq_processors = self.processors

        for frame_idx in range(total_frames):
            data = self.eeg.get_data(channel_idx, current_time)
            if data is None:
                black_frame = np.zeros((out_height, out_width), dtype=np.uint8)
                video_writer.write(black_frame)
                current_time += dt
                continue

            wavelet_processed_data = self.eeg.process_with_wavelet(data)
            band_amplitudes = self.eeg.process_bands(wavelet_processed_data)

            resolution = 128
            fractal_image = np.zeros((resolution, resolution))

            for band, weight in band_weights.items():
                state = freq_processors[band].update(band_amplitudes[band], dt)
                resized = cv2.resize(state, (resolution, resolution))
                fractal_image += weight * resized

            alpha_state = cv2.resize(freq_processors['alpha'].state, (resolution, resolution))
            beta_state  = cv2.resize(freq_processors['beta'].state,  (resolution, resolution))
            theta_state = cv2.resize(freq_processors['theta'].state, (resolution, resolution))

            alpha_beta_coupling = np.multiply(alpha_state, beta_state) * 0.1
            theta_alpha_coupling = np.multiply(theta_state, alpha_state) * 0.1
            fractal_image += alpha_beta_coupling + theta_alpha_coupling

            fractal_image = (fractal_image - fractal_image.min()) / (fractal_image.max() - fractal_image.min() + 1e-8)
            fractal_image = np.power(fractal_image, 0.85)
            fractal_image = (fractal_image * 255).astype(np.uint8)

            edge_enhanced = cv2.Laplacian(fractal_image, cv2.CV_64F).astype(np.uint8)
            fractal_image = cv2.addWeighted(fractal_image, 0.9, edge_enhanced, 0.1, 0)

            blur1 = cv2.GaussianBlur(fractal_image, (3, 3), 0)
            blur2 = cv2.GaussianBlur(fractal_image, (7, 7), 0)
            fractal_image = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

            fractal_pil = Image.fromarray(fractal_image)
            input_tensor = self.decode_transform(fractal_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output, _ = self.decoder_model(input_tensor)
                reconstructed = output.squeeze(0).cpu()

            reconstructed_image = transforms.ToPILImage()(reconstructed)
            decoded_arr = np.array(reconstructed_image)

            video_writer.write(decoded_arr)
            current_time += dt

        video_writer.release()
        messagebox.showinfo("Success", f"Video saved to: {out_file}")

    def play_video_with_model(self):
        """
        Demonstration of decoding fractal frames in real-time and displaying them in a new window.
        We'll do 5 seconds from current_time; you can adjust as needed.
        """
        if not self.eeg.raw:
            messagebox.showerror("Error", "No EEG loaded.")
            return
        if not self.decoder_model:
            messagebox.showerror("Error", "No decoder model loaded.")
            return
        
        play_win = tk.Toplevel(self.root)
        play_win.title("Playing Decoded Video")
        play_win.geometry("300x300")
        canvas_play = tk.Canvas(play_win, width=256, height=256, bg='white')
        canvas_play.pack()

        fps = 15
        duration = 5.0
        total_frames = int(duration * fps)
        dt = 1.0 / fps
        current_time = self.time_var.get()

        channel_idx = 0
        if self.channel_var.get() in self.eeg.raw.ch_names:
            channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())

        band_weights = {'theta': 0.15, 'alpha': 0.35, 'beta': 0.25, 'gamma': 0.25}
        freq_processors = self.processors
        
        def update_frame(frame_idx=0):
            if frame_idx >= total_frames:
                return

            data = self.eeg.get_data(channel_idx, current_time + frame_idx * dt)
            if data is None:
                black = np.zeros((256, 256), dtype=np.uint8)
                img = ImageTk.PhotoImage(Image.fromarray(black))
                canvas_play.create_image(0, 0, image=img, anchor=tk.NW)
                canvas_play.image = img
                play_win.after(int(1000/fps), lambda: update_frame(frame_idx+1))
                return

            wavelet_processed_data = self.eeg.process_with_wavelet(data)
            band_amplitudes = self.eeg.process_bands(wavelet_processed_data)

            resolution = 128
            fractal_image = np.zeros((resolution, resolution))

            for band, weight in band_weights.items():
                state = freq_processors[band].update(band_amplitudes[band], dt)
                resized = cv2.resize(state, (resolution, resolution))
                fractal_image += weight * resized

            alpha_state = cv2.resize(freq_processors['alpha'].state, (resolution, resolution))
            beta_state  = cv2.resize(freq_processors['beta'].state,  (resolution, resolution))
            theta_state = cv2.resize(freq_processors['theta'].state, (resolution, resolution))

            alpha_beta_coupling = np.multiply(alpha_state, beta_state) * 0.1
            theta_alpha_coupling = np.multiply(theta_state, alpha_state) * 0.1
            fractal_image += alpha_beta_coupling + theta_alpha_coupling

            fractal_image = (fractal_image - fractal_image.min()) / (fractal_image.max() - fractal_image.min() + 1e-8)
            fractal_image = np.power(fractal_image, 0.85)
            fractal_image = (fractal_image * 255).astype(np.uint8)

            edge_enhanced = cv2.Laplacian(fractal_image, cv2.CV_64F).astype(np.uint8)
            fractal_image = cv2.addWeighted(fractal_image, 0.9, edge_enhanced, 0.1, 0)

            blur1 = cv2.GaussianBlur(fractal_image, (3, 3), 0)
            blur2 = cv2.GaussianBlur(fractal_image, (7, 7), 0)
            fractal_image = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

            fractal_pil = Image.fromarray(fractal_image)
            input_tensor = self.decode_transform(fractal_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output, _ = self.decoder_model(input_tensor)
                reconstructed = output.squeeze(0).cpu()

            reconstructed_image = transforms.ToPILImage()(reconstructed)
            decoded_arr = np.array(reconstructed_image)

            img = ImageTk.PhotoImage(Image.fromarray(decoded_arr))
            canvas_play.create_image(0, 0, image=img, anchor=tk.NW)
            canvas_play.image = img

            play_win.after(int(1000/fps), lambda: update_frame(frame_idx+1))

        update_frame(0)

###############################################################################
# 4) MAIN ENTRY POINT
###############################################################################

def main():
    root = tk.Tk()
    app = Visualizer(root)
    # Make the window big enough to see all controls
    root.geometry("1200x1200")  
    root.mainloop()

if __name__ == "__main__":
    main()
