import os
import time
import logging
import hashlib
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import mne
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

# Additional scientific imports for frequency analysis
from scipy import signal

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is contiguous and has positive strides."""
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr.copy()

# -----------------------------------------------------------------------------
# Hierarchical Fractal Processor
# -----------------------------------------------------------------------------
class HierarchicalFractalProcessor:
    """
    Processes EEG information using frequency-mapped fractals.
    Maps lower frequencies (theta/alpha) to coarse structure (V1-like),
    and higher frequencies (beta/gamma) to fine details (V4/IT-like).
    """
    def __init__(self, resolution=512, fractal_dir='fractals'):
        self.resolution = resolution
        self.fractal_dir = fractal_dir
        os.makedirs(self.fractal_dir, exist_ok=True)
        
        # Define frequency bands and their properties
        self.frequency_bands = {
            'theta': {  # 4-8 Hz - coarse structure, "V1-like"
                'range': (4, 8),
                'scale_range': (0.5, 1.0),   # Larger transforms => coarser fractal
                'transforms': 3,            # Fewer transforms => simpler patterns
                'detail_weight': 0.2,       # Lower detail weighting
                'original_scale_range': (0.5, 1.0)  # Store original for scaling
            },
            'alpha': {  # 8-13 Hz - intermediate structure, "V2-like"
                'range': (8, 13),
                'scale_range': (0.3, 0.6),
                'transforms': 4,
                'detail_weight': 0.4,
                'original_scale_range': (0.3, 0.6)
            },
            'beta': {  # 13-30 Hz - fine structure, "V4-like"
                'range': (13, 30),
                'scale_range': (0.2, 0.4),
                'transforms': 5,
                'detail_weight': 0.6,
                'original_scale_range': (0.2, 0.4)
            },
            'gamma': {  # 30-100 Hz - finest details, "IT-like"
                'range': (30, 100),
                'scale_range': (0.1, 0.3),
                'transforms': 6,
                'detail_weight': 0.8,
                'original_scale_range': (0.1, 0.3)
            }
        }

    def generate_deterministic_seed(self, image_path: str) -> int:
        """Generate a deterministic seed based on the image path."""
        hash_digest = hashlib.md5(image_path.encode()).hexdigest()
        return int(hash_digest, 16) % (2**32)
    
    def analyze_eeg_frequencies(self, eeg_data: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Extract power in different frequency bands from 1D EEG data.

        :param eeg_data: 1D numpy array of the EEG signal (e.g., 1s window).
        :param fs: Sampling frequency of the EEG data.
        :return: dict of band_name -> average power
        """
        band_powers = {}
        # For each band, perform bandpass filtering and compute power
        for band_name, band_info in self.frequency_bands.items():
            low, high = band_info['range']
            nyq = fs / 2
            try:
                b, a = signal.butter(4, [low/nyq, high/nyq], btype='band', analog=False)
                # Filter signal
                filtered = signal.filtfilt(b, a, eeg_data)
                # Power calculation
                band_powers[band_name] = np.mean(filtered**2)
            except Exception as e:
                logging.error(f"Error in frequency analysis for band {band_name}: {e}")
                band_powers[band_name] = 0.0  # Assign zero power if error occurs
                
        return band_powers

    def generate_frequency_mapped_fractal(self, band_powers: Dict[str, float], seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a combined fractal image where each band influences a scale.
        :param band_powers: dict of band_name -> band power
        :param seed: Optional seed for deterministic fractal generation
        :return: fractal image (float32, range ~ [0..1])
        """
        if seed is not None:
            np.random.seed(seed)

        combined_fractal = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        total_power = sum(band_powers.values())
        if total_power < 1e-12:
            # If no power, or all-zero EEG, just return empty
            return combined_fractal

        for band_name, power in band_powers.items():
            band_info = self.frequency_bands[band_name]
            # relative power
            rel_power = power / total_power

            # 1) Generate fractal points for this band
            fractal = self._generate_band_fractal(band_info, rel_power)

            # 2) Process fractal to add details
            processed = self._process_band_fractal(fractal, band_info, rel_power)

            # 3) Weighted sum
            combined_fractal += processed * band_info['detail_weight']

        # Normalize final result
        min_val, max_val = combined_fractal.min(), combined_fractal.max()
        if (max_val - min_val) > 1e-12:
            combined_fractal = (combined_fractal - min_val) / (max_val - min_val)
        return combined_fractal

    def _generate_band_fractal(self, band_info: dict, relative_power: float) -> np.ndarray:
        """
        Generate fractal points for a single frequency band.
        """
        resolution = self.resolution
        points = np.zeros((5000, 2), dtype=np.float32)
        x, y = 0.0, 0.0

        # More or fewer transforms depending on band
        transforms = []
        scale_min, scale_max = band_info['scale_range']
        for _ in range(band_info['transforms']):
            # Scale factors
            scale = scale_min + (scale_max - scale_min) * relative_power
            rotation = np.random.uniform(0, 2*np.pi)

            c, s = np.cos(rotation), np.sin(rotation)
            tx = np.random.uniform(-0.5, 0.5)
            ty = np.random.uniform(-0.5, 0.5)
            
            transform = np.array([
                [c*scale, -s*scale, tx],
                [s*scale,  c*scale, ty],
                [0,        0,       1.0]
            ], dtype=np.float32)
            transforms.append(transform)

        # Iterated Function System (IFS)
        for i in range(len(points)):
            transform = transforms[np.random.randint(len(transforms))]
            point = np.array([x, y, 1.0], dtype=np.float32)
            transformed_point = transform @ point
            # Handle potential overflows or invalid values
            if not np.isfinite(transformed_point).all():
                x, y = 0.0, 0.0  # Reset to origin on invalid transformation
            else:
                x, y = transformed_point[0], transformed_point[1]
            points[i] = [x, y]

        # Clip points to prevent extreme values
        points = np.clip(points, -1.0, 1.0)

        # Map to [0..1]
        points = (points + 1.0) / 2.0
        # Scale to resolution
        coords = (points * (resolution - 1)).astype(int)

        # Render to image
        image = np.zeros((resolution, resolution), dtype=np.float32)
        valid_mask = (
            (coords[:, 0] >= 0) & (coords[:, 0] < resolution) &
            (coords[:, 1] >= 0) & (coords[:, 1] < resolution)
        )
        valid_points = coords[valid_mask]

        # Increase intensity
        for px, py in valid_points:
            image[py, px] += 1.0

        # Normalize
        if image.max() > 1e-12:
            image = image / image.max()
        return image

    def _process_band_fractal(self, fractal: np.ndarray, band_info: dict, relative_power: float) -> np.ndarray:
        """
        Optionally apply multi-scale or blur-based 'detail' processing.
        """
        # Create a small Gaussian pyramid to extract details
        levels = 3
        pyramid = []
        current = fractal.copy()

        for _ in range(levels):
            h, w = current.shape
            if min(h, w) < 2:
                break
            scaled = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            pyramid.append(scaled)
            current = scaled

        # Reconstruct with frequency-dependent weighting
        processed = np.zeros_like(fractal)
        size = fractal.shape[::-1]  # (width, height)
        for i, level_img in enumerate(pyramid):
            # Higher-level pyramid = finer detail
            weight = relative_power * (1.0 - i / len(pyramid))
            up = cv2.resize(level_img, size, interpolation=cv2.INTER_LINEAR)
            processed += up * weight

        # Combine with the base fractal
        processed += fractal * 0.5

        # Final normalize
        mx = processed.max()
        if mx > 1e-12:
            processed /= mx

        return processed

    def generate_and_save_fractal(self, eeg_data: np.ndarray, fs: float, seed: Optional[int] = None) -> str:
        """
        Generate fractal from EEG data and save it.
        :param eeg_data: 1D numpy array of EEG data.
        :param fs: Sampling frequency.
        :param seed: Optional seed for deterministic fractal generation.
        :return: Path to the saved fractal image.
        """
        band_powers = self.analyze_eeg_frequencies(eeg_data, fs)
        fractal = self.generate_frequency_mapped_fractal(band_powers, seed)
        fractal_uint8 = (fractal * 255).astype(np.uint8)
        fractal_pil = Image.fromarray(fractal_uint8)
        # Generate a unique filename based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fractal_filename = f"fractal_generated_{timestamp}.png"
        fractal_path = os.path.join(self.fractal_dir, fractal_filename)
        fractal_pil.save(fractal_path)
        logging.info(f"Generated and saved fractal: {fractal_path}")
        return fractal_path

    def save_fractal_from_original(self, original_image_path: str, band_powers: Dict[str, float]) -> str:
        """
        Generate and save the fractal image corresponding to the original image.
        :param original_image_path: Path to the original image.
        :param band_powers: Band powers used for fractal generation.
        :return: Path to the saved fractal image.
        """
        fractal = self.generate_frequency_mapped_fractal(band_powers)
        fractal_uint8 = (fractal * 255).astype(np.uint8)
        fractal_pil = Image.fromarray(fractal_uint8)
        original_basename = os.path.basename(original_image_path)
        if original_basename.startswith('fractal_'):
            fractal_filename = original_basename  # Avoid double prefixing
        else:
            fractal_filename = f"fractal_{original_basename}"
        fractal_path = os.path.join(self.fractal_dir, fractal_filename)
        fractal_pil.save(fractal_path)
        logging.info(f"Saved fractal image: {fractal_path}")
        return fractal_path

# -----------------------------------------------------------------------------
# EEG Processing
# -----------------------------------------------------------------------------
class EEGProcessor:
    """Handles EEG data loading and retrieval."""
    def __init__(self, resolution: int = 512):
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        # 1 second window for retrieval
        self.window_size = 1.0  
        self.resolution = resolution

    def load_file(self, filepath: str) -> bool:
        """Load EEG data from an EDF file."""
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            return True
        except Exception as e:
            logging.error(f"Failed to load EEG file: {e}")
            return False

    def get_channels(self):
        """Return list of channel names."""
        if self.raw:
            return self.raw.ch_names
        return []

    def get_data(self, channel: int, start_time: float) -> Optional[np.ndarray]:
        """Retrieve 1s of EEG data for a specific channel and time."""
        if self.raw is None:
            return None
        try:
            start_sample = int(start_time * self.sfreq)
            samples_needed = int(self.window_size * self.sfreq)
            end_sample = start_sample + samples_needed
            if end_sample > self.raw.n_times:
                end_sample = self.raw.n_times
            data, _ = self.raw[channel, start_sample:end_sample]
            return data.flatten()
        except Exception as e:
            logging.error(f"Error getting EEG data: {e}")
            return None

# -----------------------------------------------------------------------------
# Enhanced U-Net Model
# -----------------------------------------------------------------------------
class EnhancedUNet(nn.Module):
    """U-Net for reversing fractal transformations (outputs grayscale)."""
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

        # Final conv => 1 channel
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        """Helper for double convolution layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.final_conv(d1)

# -----------------------------------------------------------------------------
# Dataset for Training
# -----------------------------------------------------------------------------
class ImagePairDataset(Dataset):
    """Dataset for image pairs (fractal and original)."""
    def __init__(self, original_dir, fractal_dir, image_pairs, transform=None):
        self.original_dir = original_dir
        self.fractal_dir = fractal_dir
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        frac_file, orig_file = self.image_pairs[idx]  # Note the order: fractal first, then original

        # Load images
        fractal_img = Image.open(os.path.join(self.fractal_dir, frac_file)).convert('L')
        original_img = Image.open(os.path.join(self.original_dir, orig_file)).convert('L')

        if self.transform:
            fractal_img = self.transform(fractal_img)
            original_img = self.transform(original_img)

        return fractal_img, original_img

# -----------------------------------------------------------------------------
# Decoder Testing
# -----------------------------------------------------------------------------
class DecoderTest:
    """Testing framework for fractal-to-image decoding."""
    def __init__(self, model, device, resolution=512, fractal_dir='fractals'):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.fractal_dir = fractal_dir  # Directory where fractals are stored
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        # We'll create one fractal processor for testing
        self.fractal_processor = HierarchicalFractalProcessor(resolution=self.resolution, fractal_dir=self.fractal_dir)

    def process_paired_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Test fractal encoding and decoding on a paired image (original and fractal).
        Returns: (original, fractal, decoded, PSNR, SSIM)
        """
        # Load and preprocess original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Invalid image or cannot open: {image_path}")

        # Resize to match the resolution
        original = cv2.resize(original, (self.resolution, self.resolution))

        # Load the corresponding fractal image
        # Assuming fractal images are named as 'fractal_<original_filename>'
        original_basename = os.path.basename(image_path)
        if original_basename.startswith('fractal_'):
            fractal_filename = original_basename  # Avoid double prefixing
        else:
            fractal_filename = f"fractal_{original_basename}"
        fractal_path = os.path.join(self.fractal_dir, fractal_filename)
        fractal = cv2.imread(fractal_path, cv2.IMREAD_GRAYSCALE)
        if fractal is None:
            raise ValueError(f"Corresponding fractal image not found: {fractal_path}")

        # Resize fractal to match resolution if necessary
        fractal = cv2.resize(fractal, (self.resolution, self.resolution))

        # Decode fractal
        fractal_tensor = self.transform(Image.fromarray(fractal)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            decoded_tensor = self.model(fractal_tensor)
            decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
            if decoded_np.shape[0] > 1:
                logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
            decoded_np = decoded_np[0]  # channel 0
            decoded_np = (decoded_np * 255).astype(np.uint8)
            decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        # Metrics
        if original.shape != decoded_np.shape:
            raise ValueError(
                f"Shape mismatch: original={original.shape}, decoded={decoded_np.shape}. "
                "They must be identical for PSNR/SSIM."
            )

        psnr_val = psnr(original, decoded_np)
        ssim_val = ssim(original, decoded_np)

        return original, fractal, decoded_np, psnr_val, ssim_val

    def decode_any_fractal_image(self, fractal_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode any standalone fractal image without needing a paired original.
        Returns: (fractal, decoded)
        """
        fractal = cv2.imread(fractal_path, cv2.IMREAD_GRAYSCALE)
        if fractal is None:
            raise ValueError(f"Invalid fractal image or cannot open: {fractal_path}")

        # Resize fractal to match resolution if necessary
        fractal = cv2.resize(fractal, (self.resolution, self.resolution))

        # Decode fractal
        fractal_tensor = self.transform(Image.fromarray(fractal)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            decoded_tensor = self.model(fractal_tensor)
            decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
            if decoded_np.shape[0] > 1:
                logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
            decoded_np = decoded_np[0]  # channel 0
            decoded_np = (decoded_np * 255).astype(np.uint8)
            decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        return fractal, decoded_np

# -----------------------------------------------------------------------------
# Video Recording
# -----------------------------------------------------------------------------
class VideoRecorder:
    """Handles recording of EEG visualization to video."""
    def __init__(self, resolution=512):
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        self.output_path = None

    def start_recording(self, output_path: str, fps: float = 30.0):
        """Start recording video."""
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.resolution * 2, self.resolution)
        )
        self.is_recording = True
        logging.info(f"Started recording to {output_path}")

    def add_frame(self, fractal_frame: np.ndarray, decoded_frame: np.ndarray):
        """Add a frame to the video (side-by-side)."""
        if not self.is_recording:
            return

        fractal_frame = cv2.resize(fractal_frame, (self.resolution, self.resolution))
        decoded_frame = cv2.resize(decoded_frame, (self.resolution, self.resolution))
        combined = np.hstack([fractal_frame, decoded_frame])
        self.writer.write(combined)

    def stop_recording(self):
        """Stop recording and save video."""
        if self.writer:
            self.writer.release()
        self.is_recording = False
        self.writer = None
        logging.info(f"Stopped recording and saved to {self.output_path}")

# -----------------------------------------------------------------------------
# Results Logger
# -----------------------------------------------------------------------------
class ResultsLogger:
    """Logs and tracks decoder performance metrics."""
    def __init__(self, log_dir: str = "decoder_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []

    def log_result_paired(self, original_path: str, psnr_val: float, ssim_val: float):
        """Log metrics for a paired decode attempt."""
        self.metrics.append({
            'type': 'Paired',
            'image': original_path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'timestamp': time.time()
        })
        logging.info(f"Logged paired results for {original_path}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

    def log_result_standalone(self, fractal_path: str, decoded_image: np.ndarray):
        """Log metrics for a standalone decode attempt."""
        # Since there's no original image, metrics like PSNR and SSIM are not applicable
        self.metrics.append({
            'type': 'Standalone',
            'image': fractal_path,
            'decoded': decoded_image,
            'timestamp': time.time()
        })
        logging.info(f"Logged standalone decode for {fractal_path}")

    def save_metrics(self):
        """Save all metrics to file."""
        if not self.metrics:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(self.log_dir, f"metrics_{timestamp}.txt")
        with open(log_path, 'w') as f:
            for m in self.metrics:
                if m['type'] == 'Paired':
                    f.write(f"{m['type']},{m['image']},{m['psnr']:.4f},{m['ssim']:.4f}\n")
                elif m['type'] == 'Standalone':
                    f.write(f"{m['type']},{m['image']},Decoded Image Saved\n")
        logging.info(f"Saved metrics to {log_path}")

    def get_summary(self) -> dict:
        """Get summary statistics of all paired metrics."""
        psnr_vals = [m['psnr'] for m in self.metrics if m['type'] == 'Paired']
        ssim_vals = [m['ssim'] for m in self.metrics if m['type'] == 'Paired']

        if not psnr_vals:
            return {'psnr_avg': 0, 'ssim_avg': 0, 'psnr_std': 0, 'ssim_std': 0}

        return {
            'psnr_avg': np.mean(psnr_vals),
            'psnr_std': np.std(psnr_vals),
            'ssim_avg': np.mean(ssim_vals),
            'ssim_std': np.std(ssim_vals)
        }

# -----------------------------------------------------------------------------
# Visualization State
# -----------------------------------------------------------------------------
@dataclass
class VisualizationState:
    """Holds the current state of visualization."""
    current_time: float = 0.0
    is_playing: bool = False
    seek_requested: bool = False
    seek_time: float = 0.0
    playback_speed: float = 1.0

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
class FractalEEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractal EEG Analyzer")

        # Initialize components
        self.resolution = 512
        self.eeg = EEGProcessor(resolution=self.resolution)
        self.state = VisualizationState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedUNet().to(self.device)
        self.model.eval()  # default to eval mode

        self.last_update = time.time()

        # Initialize fractal processor with a specified fractal directory
        self.fractal_dir = 'fractals'
        self.fractal_processor = HierarchicalFractalProcessor(resolution=self.resolution, fractal_dir=self.fractal_dir)

        # Additional components
        self.decoder_test = DecoderTest(self.model, self.device, self.resolution, fractal_dir=self.fractal_dir)
        self.video_recorder = VideoRecorder(self.resolution)
        self.results_logger = ResultsLogger()

        self.setup_gui()

    def setup_gui(self):
        """Set up the graphical user interface."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG", command=self.load_eeg)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Testing menu
        test_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Testing", menu=test_menu)
        test_menu.add_command(label="Test Paired Image", command=self.test_paired_image)
        test_menu.add_command(label="Decode Fractal Image", command=self.decode_fractal_image)
        test_menu.add_command(label="Test Batch Images", command=self.test_batch_images)
        test_menu.add_command(label="View Test Results", command=self.view_test_results)

        # Processing menu
        process_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processing", menu=process_menu)
        process_menu.add_command(label="Batch Process Images", command=self.batch_process)
        process_menu.add_command(label="Train Model", command=self.train_model)

        # Recording menu
        record_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Recording", menu=record_menu)
        record_menu.add_command(label="Start Recording", command=self.start_recording)
        record_menu.add_command(label="Stop Recording", command=self.stop_recording)

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel
        control_frame = ttk.LabelFrame(main_container, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(control_frame, text="EEG Channel:").pack(pady=5)
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var)
        self.channel_combo.pack(fill=tk.X, padx=5, pady=5)

        play_frame = ttk.Frame(control_frame)
        play_frame.pack(fill=tk.X, pady=5)
        self.play_btn = ttk.Button(play_frame, text="Play", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Time Position (s):").pack(pady=5)
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(control_frame, from_=0, to=100,
                                     variable=self.time_var, command=self.seek)
        self.time_slider.pack(fill=tk.X, padx=5, pady=5)

        # Frequency Controls
        self.add_frequency_controls(control_frame)

        # Right visualization panel
        viz_frame = ttk.LabelFrame(main_container, text="Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        fractal_frame = ttk.Frame(viz_frame)
        fractal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(fractal_frame, text="Fractal Pattern").pack()
        self.fractal_canvas = tk.Canvas(fractal_frame, bg='black', width=512, height=512)
        self.fractal_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        decoded_frame = ttk.Frame(viz_frame)
        decoded_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(decoded_frame, text="Decoded Image").pack()
        self.decoded_canvas = tk.Canvas(decoded_frame, bg='black', width=512, height=512)
        self.decoded_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # -------------------------------------------------------------------------
    # GUI Command Methods
    # -------------------------------------------------------------------------
    def load_eeg(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if filepath and self.eeg.load_file(filepath):
            self.channel_combo['values'] = self.eeg.get_channels()
            self.channel_combo.set(self.eeg.get_channels()[0])
            self.time_slider.configure(to=self.eeg.duration)
            messagebox.showinfo("Success", "EEG file loaded successfully")
            logging.info(f"EEG file loaded: {filepath}")
            self.update_visualization()
        else:
            messagebox.showerror("Error", "Failed to load EEG file")
            logging.error("Failed to load EEG file")

    def toggle_playback(self):
        self.state.is_playing = not self.state.is_playing
        self.play_btn.configure(text="Pause" if self.state.is_playing else "Play")
        if self.state.is_playing:
            self.last_update = time.time()
            self.update()

    def seek(self, value):
        """When user drags the time slider."""
        if not self.state.is_playing:
            self.state.seek_requested = True
            self.state.seek_time = float(value)
            self.state.current_time = float(value)
            self.update_visualization()

    def update(self):
        """Called repeatedly during playback."""
        if not self.state.is_playing:
            return

        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        self.state.current_time += dt * self.state.playback_speed

        # Loop around if at end
        if self.state.current_time >= self.eeg.duration:
            self.state.current_time = 0

        self.time_var.set(self.state.current_time)
        self.update_visualization()
        self.root.after(33, self.update)  # ~30 FPS

    def update_visualization(self):
        """Compute fractal from current EEG snippet and decode it."""
        if not self.eeg.raw or not self.channel_var.get():
            return
        try:
            channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
            data = self.eeg.get_data(channel_idx, self.state.current_time)
            if data is None:
                return

            # data is 1s. Generate fractal
            fractal_path = self.fractal_processor.generate_and_save_fractal(data, self.eeg.sfreq)

            # Load fractal image
            fractal = cv2.imread(fractal_path, cv2.IMREAD_GRAYSCALE)
            if fractal is None:
                logging.error(f"Failed to load generated fractal: {fractal_path}")
                return

            fractal = cv2.resize(fractal, (self.resolution, self.resolution))

            # Apply color map for visualization
            colored = cv2.applyColorMap(fractal, cv2.COLORMAP_JET)

            # Show fractal in the left canvas
            fractal_pil = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
            fractal_pil = fractal_pil.resize((512, 512), Image.LANCZOS)
            fractal_photo = ImageTk.PhotoImage(image=fractal_pil)
            self.fractal_canvas.delete("all")
            self.fractal_canvas.create_image(0, 0, image=fractal_photo, anchor=tk.NW)
            self.fractal_canvas.image = fractal_photo

            # Decode fractal back to grayscale image
            fractal_tensor = self.decoder_test.transform(
                Image.fromarray(fractal)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                decoded_tensor = self.model(fractal_tensor)
                decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
                if decoded_np.shape[0] > 1:
                    logging.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
                decoded_np = decoded_np[0]  # channel 0
                decoded_np = (decoded_np * 255).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

            decoded_image = Image.fromarray(decoded_np)
            decoded_photo = ImageTk.PhotoImage(image=decoded_image)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            self.decoded_canvas.image = decoded_photo

            # If we're recording, add frame
            if self.video_recorder.is_recording:
                # Convert decoded_np to BGR for consistency
                decoded_bgr = cv2.cvtColor(decoded_np, cv2.COLOR_GRAY2BGR)
                self.video_recorder.add_frame(colored, decoded_bgr)

        except Exception as e:
            logging.error(f"Error in visualization update: {e}")

    def add_frequency_controls(self, ctrl_frame):
        """Add frequency band control sliders to GUI"""
        freq_frame = ttk.LabelFrame(ctrl_frame, text="Frequency Controls")
        freq_frame.pack(fill=tk.X, padx=5, pady=5)

        self.freq_controls = {}

        # For each frequency band, add power, coupling, and frequency sliders
        for band in ['theta', 'alpha', 'beta', 'gamma']:
            band_frame = ttk.LabelFrame(freq_frame, text=f"{band.title()}")
            band_frame.pack(fill=tk.X, padx=5, pady=2)

            # Power control
            ttk.Label(band_frame, text="Power:").pack(anchor='w')
            power_var = tk.DoubleVar(value=1.0)
            power_slider = ttk.Scale(
                band_frame, from_=0.0, to=2.0,
                variable=power_var,
                command=lambda v, b=band: self.update_band_power(b, float(v))
            )
            power_slider.pack(fill=tk.X, padx=5, pady=2)
            power_slider.set(1.0)  # Reset to default

            # Coupling control
            ttk.Label(band_frame, text="Coupling:").pack(anchor='w')
            coupling_var = tk.DoubleVar(value=1.0)
            coupling_slider = ttk.Scale(
                band_frame, from_=0.0, to=2.0,
                variable=coupling_var,
                command=lambda v, b=band: self.update_band_coupling(b, float(v))
            )
            coupling_slider.pack(fill=tk.X, padx=5, pady=2)
            coupling_slider.set(1.0)  # Reset to default

            # Oscillation frequency control
            ttk.Label(band_frame, text="Frequency (Hz):").pack(anchor='w')
            freq_var = tk.DoubleVar(value=(
                self.fractal_processor.frequency_bands[band]['range'][0] + 
                self.fractal_processor.frequency_bands[band]['range'][1]
            ) / 2)
            freq_slider = ttk.Scale(
                band_frame, 
                from_=self.fractal_processor.frequency_bands[band]['range'][0],
                to=self.fractal_processor.frequency_bands[band]['range'][1],
                variable=freq_var,
                command=lambda v, b=band: self.update_band_frequency(b, float(v))
            )
            freq_slider.pack(fill=tk.X, padx=5, pady=2)
            freq_slider.set(freq_var.get())  # Reset to default

            self.freq_controls[band] = {
                'power': power_var,
                'coupling': coupling_var,
                'frequency': freq_var
            }

    def update_band_power(self, band, value):
        """Update band power scaling"""
        # Reset to original scale ranges
        original_scale_min, original_scale_max = self.fractal_processor.frequency_bands[band]['original_scale_range']
        # Apply scaling based on slider value
        new_scale_min = original_scale_min * value
        new_scale_max = original_scale_max * value
        self.fractal_processor.frequency_bands[band]['scale_range'] = (new_scale_min, new_scale_max)
        logging.info(f"Updated power scaling for {band} band to {value}")

        # Update visualization
        self.update_visualization()

    def update_band_coupling(self, band, value):
        """Update coupling strength for band"""
        # Here, coupling could influence the detail_weight
        # Ensure that detail_weight stays within [0,1] after scaling
        original_detail_weight = self.fractal_processor.frequency_bands[band]['detail_weight'] / 2  # Assuming original was set with scale=1.0
        new_detail_weight = original_detail_weight * value
        new_detail_weight = min(max(new_detail_weight, 0.0), 1.0)
        self.fractal_processor.frequency_bands[band]['detail_weight'] = new_detail_weight
        logging.info(f"Updated coupling for {band} band to {value}, new detail_weight: {new_detail_weight}")

        # Update visualization
        self.update_visualization()

    def update_band_frequency(self, band, value):
        """Update oscillation frequency"""
        # Update the frequency range based on the center frequency
        band_info = self.fractal_processor.frequency_bands[band]
        center_freq = value
        bandwidth = (band_info['range'][1] - band_info['range'][0]) / 2
        new_low = max(center_freq - bandwidth / 2, 0.1)  # Prevent negative frequencies
        new_high = center_freq + bandwidth / 2
        if new_high <= new_low:
            new_high = new_low + 0.1  # Ensure high > low

        # Update the frequency range
        band_info['range'] = (new_low, new_high)
        logging.info(f"Updated frequency range for {band} band to {new_low:.2f} - {new_high:.2f} Hz")

        # Update visualization
        self.update_visualization()

    # -------------------------------------------------------------------------
    # Testing / Batch
    # -------------------------------------------------------------------------
    def test_paired_image(self):
        """Test decoding on a paired original-fractal image."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            results_window = tk.Toplevel(self.root)
            results_window.title("Decoder Test Results (Paired)")

            original, fractal, decoded, psnr_val, ssim_val = self.decoder_test.process_paired_image(filepath)

            images_frame = ttk.Frame(results_window)
            images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Original Image
            original_frame = ttk.LabelFrame(images_frame, text="Original Image")
            original_frame.pack(side=tk.LEFT, padx=5)
            original_canvas = tk.Canvas(original_frame, width=256, height=256)
            original_canvas.pack()
            original_pil = Image.fromarray(original)
            original_pil = original_pil.resize((256, 256), Image.LANCZOS)
            original_photo = ImageTk.PhotoImage(original_pil)
            original_canvas.create_image(0, 0, image=original_photo, anchor=tk.NW)
            original_canvas.image = original_photo

            # Fractal Image
            fractal_frame = ttk.LabelFrame(images_frame, text="Fractal")
            fractal_frame.pack(side=tk.LEFT, padx=5)
            fractal_canvas = tk.Canvas(fractal_frame, width=256, height=256)
            fractal_canvas.pack()
            fractal_pil = Image.fromarray(fractal)
            fractal_pil = fractal_pil.resize((256, 256), Image.LANCZOS)
            fractal_photo = ImageTk.PhotoImage(fractal_pil)
            fractal_canvas.create_image(0, 0, image=fractal_photo, anchor=tk.NW)
            fractal_canvas.image = fractal_photo

            # Decoded Image
            decoded_frame = ttk.LabelFrame(images_frame, text="Decoded Image")
            decoded_frame.pack(side=tk.LEFT, padx=5)
            decoded_canvas = tk.Canvas(decoded_frame, width=256, height=256)
            decoded_canvas.pack()
            decoded_pil = Image.fromarray(decoded)
            decoded_pil = decoded_pil.resize((256, 256), Image.LANCZOS)
            decoded_photo = ImageTk.PhotoImage(decoded_pil)
            decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            decoded_canvas.image = decoded_photo

            # Metrics
            metrics_frame = ttk.Frame(results_window)
            metrics_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(metrics_frame, text=f"PSNR: {psnr_val:.2f} dB").pack(side=tk.LEFT, padx=10)
            ttk.Label(metrics_frame, text=f"SSIM: {ssim_val:.4f}").pack(side=tk.LEFT, padx=10)

            # Log results
            self.results_logger.log_result_paired(filepath, psnr_val, ssim_val)

        except Exception as e:
            logging.error(f"Error in paired decoder test: {e}")
            messagebox.showerror("Error", f"Paired Test failed: {str(e)}")

    def decode_fractal_image(self):
        """Decode any standalone fractal image without needing a paired original."""
        fractal_path = filedialog.askopenfilename(
            title="Select Fractal Image to Decode",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not fractal_path:
            return
        try:
            # Decode fractal
            fractal, decoded = self.decoder_test.decode_any_fractal_image(fractal_path)

            # Show fractal in the left canvas
            fractal_colored = cv2.applyColorMap(fractal, cv2.COLORMAP_JET)
            fractal_pil = Image.fromarray(cv2.cvtColor(fractal_colored, cv2.COLOR_BGR2RGB))
            fractal_pil = fractal_pil.resize((512, 512), Image.LANCZOS)
            fractal_photo = ImageTk.PhotoImage(image=fractal_pil)
            self.fractal_canvas.delete("all")
            self.fractal_canvas.create_image(0, 0, image=fractal_photo, anchor=tk.NW)
            self.fractal_canvas.image = fractal_photo

            # Show decoded image in the right canvas
            decoded_pil = Image.fromarray(decoded)
            decoded_pil = decoded_pil.resize((512, 512), Image.LANCZOS)
            decoded_photo = ImageTk.PhotoImage(image=decoded_pil)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            self.decoded_canvas.image = decoded_photo

            # Log standalone decode
            self.results_logger.log_result_standalone(fractal_path, decoded)

            # If we're recording, add frame
            if self.video_recorder.is_recording:
                # Convert decoded to BGR for consistency
                decoded_bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
                self.video_recorder.add_frame(fractal_colored, decoded_bgr)

            # Inform user
            messagebox.showinfo("Success", f"Decoded fractal image:\n{fractal_path}")

        except Exception as e:
            logging.error(f"Error in decoding fractal image: {e}")
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")

    def test_batch_images(self):
        input_dir = filedialog.askdirectory(title="Select Directory with Test Images")
        if not input_dir:
            return
        try:
            image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not image_files:
                messagebox.showwarning("No Images", "No images found in directory")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Testing Progress")
            progress_window.geometry("300x150")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Testing images...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            for i, filename in enumerate(image_files):
                path_ = os.path.join(input_dir, filename)
                try:
                    original, fractal, decoded, psnr_val, ssim_val = self.decoder_test.process_paired_image(path_)
                    self.results_logger.log_result_paired(path_, psnr_val, ssim_val)
                except Exception as e:
                    logging.error(f"Error processing {path_}: {e}")
                    continue

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            self.results_logger.save_metrics()
            progress_window.destroy()
            messagebox.showinfo("Success", "Batch testing complete!")

        except Exception as e:
            logging.error(f"Error in batch testing: {e}")
            messagebox.showerror("Error", f"Batch testing failed: {str(e)}")

    def view_test_results(self):
        summary = self.results_logger.get_summary()
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Test Results Summary")

        ttk.Label(summary_window, text=f"Average PSNR: {summary['psnr_avg']:.2f} dB").pack(pady=5)
        ttk.Label(summary_window, text=f"PSNR Std Dev: {summary['psnr_std']:.2f} dB").pack(pady=5)
        ttk.Label(summary_window, text=f"Average SSIM: {summary['ssim_avg']:.4f}").pack(pady=5)
        ttk.Label(summary_window, text=f"SSIM Std Dev: {summary['ssim_std']:.4f}").pack(pady=5)

        results_frame = ttk.Frame(summary_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        results_listbox = tk.Listbox(results_frame, yscrollcommand=scrollbar.set, width=100)
        for metric in self.results_logger.metrics:
            if metric['type'] == 'Paired':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
                results_listbox.insert(
                    tk.END,
                    f"{timestamp} | {os.path.basename(metric['image'])} | "
                    f"PSNR: {metric['psnr']:.2f} dB | SSIM: {metric['ssim']:.4f}"
                )
            elif metric['type'] == 'Standalone':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
                results_listbox.insert(
                    tk.END,
                    f"{timestamp} | Standalone Decode | {os.path.basename(metric['image'])} | Decoded Image"
                )
        results_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=results_listbox.yview)

    def batch_process(self):
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        try:
            image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if not image_files:
                messagebox.showwarning("No Images", "No images found in input directory")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Batch Processing Progress")
            progress_window.geometry("300x150")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Processing images...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            for i, filename in enumerate(image_files):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = image.astype(float) / 255.0

                # For demonstration: compute band powers (assuming uniform band powers)
                band_powers = {
                    'theta': 1.0,
                    'alpha': 0.5,
                    'beta': 1.5,
                    'gamma': 2.0
                }
                # Generate and save fractal
                fractal_path = self.fractal_processor.save_fractal_from_original(image_path, band_powers)

                # Load the saved fractal
                if os.path.basename(fractal_path).startswith('fractal_'):
                    fractal_filename = os.path.basename(fractal_path)
                else:
                    fractal_filename = f"fractal_{os.path.basename(fractal_path)}"
                fractal_path = os.path.join(self.fractal_dir, fractal_filename)
                fractal = cv2.imread(fractal_path, cv2.IMREAD_GRAYSCALE)
                if fractal is None:
                    continue
                fractal = cv2.resize(fractal, (self.resolution, self.resolution))

                # Apply color map for visualization or other purposes
                colored = cv2.applyColorMap(fractal, cv2.COLORMAP_JET)

                # Save colored fractal to output directory
                output_path = os.path.join(output_dir, fractal_filename)
                cv2.imwrite(output_path, colored)

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", "Batch processing complete!")

        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")

    def start_recording(self):
        if self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Already recording!")
            return
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not output_path:
            return
        try:
            self.video_recorder.start_recording(output_path)
            messagebox.showinfo("Recording", f"Recording started: {output_path}")
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        if not self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Not currently recording!")
            return
        try:
            self.video_recorder.stop_recording()
            messagebox.showinfo("Recording", "Recording stopped and saved.")
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                messagebox.showinfo("Success", "Model loaded successfully!")
                logging.info(f"Model loaded from {filepath}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def save_model(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                torch.save({'model_state_dict': self.model.state_dict()}, filepath)
                messagebox.showinfo("Success", "Model saved successfully!")
                logging.info(f"Model saved to {filepath}")
            except Exception as e:
                logging.error(f"Error saving model: {e}")
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")


    def train_model(self):
        """Train the U-Net model on original vs fractal images."""
        original_dir = filedialog.askdirectory(title="Select Original Images Directory")
        if not original_dir:
            return
        fractal_dir = filedialog.askdirectory(title="Select Fractal Images Directory")
        if not fractal_dir:
            return

        try:
            original_files = [
                f for f in os.listdir(original_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            # Assuming fractal images are named as 'fractal_<original_filename>'
            fractal_files = [f"fractal_{f}" for f in original_files]

            valid_pairs = []
            for orig, frac in zip(original_files, fractal_files):
                if os.path.exists(os.path.join(fractal_dir, frac)):
                    valid_pairs.append((frac, orig))  # Order: fractal first, then original

            if not valid_pairs:
                messagebox.showerror("Error", "No matching image pairs found")
                return

            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training Progress")
            progress_window.geometry("400x200")

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=20)

            status_var = tk.StringVar(value="Preparing training...")
            status_label = ttk.Label(progress_window, textvariable=status_var)
            status_label.pack(pady=10)

            num_epochs = 400
            batch_size = 4
            learning_rate = 0.001

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            split_idx = int(0.8 * len(valid_pairs))
            train_pairs = valid_pairs[:split_idx]
            val_pairs = valid_pairs[split_idx:]

            train_dataset = ImagePairDataset(original_dir, fractal_dir, train_pairs, transform)
            val_dataset = ImagePairDataset(original_dir, fractal_dir, val_pairs, transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')

            for epoch in range(num_epochs):
                train_loss = 0.0
                self.model.train()

                for batch_idx, (fractal_imgs, original_imgs) in enumerate(train_loader):
                    fractal_imgs = fractal_imgs.to(self.device)
                    original_imgs = original_imgs.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(fractal_imgs)
                    loss = criterion(outputs, original_imgs)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    total_batches = len(train_loader)
                    batch_progress = 100 * (batch_idx + 1) / total_batches
                    progress_var.set((epoch + batch_idx / total_batches) / num_epochs * 100)
                    status_var.set(f"Epoch {epoch+1}/{num_epochs} - Training...")
                    progress_window.update()

                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for fractal_imgs, original_imgs in val_loader:
                        fractal_imgs = fractal_imgs.to(self.device)
                        original_imgs = original_imgs.to(self.device)
                        outputs = self.model(fractal_imgs)
                        val_loss += criterion(outputs, original_imgs).item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss
                    }, 'best_model.pth')
                    logging.info(f"Saved best model with Val Loss: {val_loss:.4f}")

                status_var.set(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", f"Training complete!\nBest validation loss: {best_val_loss:.4f}")

        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to train: {str(e)}")

 
# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    # Configure logging
    logging.basicConfig(
        filename='fractal_eeg_app.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    root = tk.Tk()
    app = FractalEEGApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()
