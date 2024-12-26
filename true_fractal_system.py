import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
from PIL import Image, ImageTk
import mne
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Supporting Classes Definitions
# ------------------------------

class FractalProcessor:
    """Handles fractal-based processing for both EEG and images using IFS"""
    def __init__(self, freq_band: tuple, resolution: int = 512):
        self.resolution = resolution
        self.freq_band = freq_band
        self.fractal_dim = 2.0

        # Parameters for IFS
        self.num_iterations = 5000
        self.num_functions = 4

        # Control parameters
        self.coupling_strength = 0.3
        self.wave_speed = 1.0
        self.signal_strength = 50.0  # 0-100 scale

        # Initialize transformation matrices
        self.transforms = self._initialize_transforms()

    def _initialize_transforms(self):
        """Initialize IFS transformation matrices"""
        transforms = []
        for _ in range(self.num_functions):
            a = np.random.uniform(-0.5, 0.5)
            b = np.random.uniform(-0.5, 0.5)
            c = np.random.uniform(-0.5, 0.5)
            d = np.random.uniform(-0.5, 0.5)
            e = np.random.uniform(-1, 1)
            f = np.random.uniform(-1, 1)

            transform = np.array([[a, b, e],
                                  [c, d, f],
                                  [0, 0, 1]])
            transforms.append(transform)
        return transforms

    def generate_fractal(self, amplitude: float) -> np.ndarray:
        """Generate fractal pattern based on amplitude and parameters"""
        # Create output image at full resolution
        image = np.zeros((self.resolution, self.resolution))
        points = np.zeros((self.num_iterations, 2))
        x, y = 0, 0

        # Scale parameters
        scale_factor = (amplitude * self.signal_strength / 100.0)

        # Generate points using IFS
        for i in range(self.num_iterations):
            transform = self.transforms[np.random.randint(self.num_functions)]
            point = np.dot(transform, np.array([x, y, 1]))
            x, y = point[0], point[1]
            points[i] = [x, y]

        # Convert points to image coordinates
        points = (points + 1) / 2  # Normalize to [0,1]
        points = (points * (self.resolution - 1)).astype(int)

        # Only use valid points
        mask = (
            (points[:, 0] >= 0) & (points[:, 0] < self.resolution) &
            (points[:, 1] >= 0) & (points[:, 1] < self.resolution)
        )
        valid_points = points[mask]

        # Place points in image
        for px, py in valid_points:
            image[py, px] += scale_factor

        # Apply wave speed modulation and smoothing
        image = cv2.GaussianBlur(image, (5, 5), 0) * self.wave_speed
        image = image / (np.max(image) + 1e-8)  # Normalize

        return image

class EEGProcessor:
    """Handles EEG data processing using fractal analysis"""
    def __init__(self, resolution: int = 512):
        self.freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        self.window_size = 1.0  # seconds
        self.resolution = resolution

        # Initialize fractal processors with consistent resolution
        self.processors = {
            band: FractalProcessor(freq_range, resolution=self.resolution)
            for band, freq_range in self.freq_bands.items()
        }

    def load_file(self, filepath: str) -> bool:
        """Load EEG data from an EDF file"""
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            return True
        except Exception as e:
            logger.error(f"Failed to load EEG file: {e}")
            return False

    def get_data(self, channel: int, start_time: float) -> Optional[np.ndarray]:
        """Retrieve EEG data for a specific channel and time"""
        if self.raw is None:
            return None
        try:
            start_sample = int(start_time * self.sfreq)
            samples_needed = int(self.window_size * self.sfreq)
            data, _ = self.raw[channel, start_sample:start_sample + samples_needed]
            return data[0]
        except Exception as e:
            logger.error(f"Error getting EEG data: {e}")
            return None

@dataclass
class VisualizationState:
    """Holds the current state of visualization"""
    current_time: float = 0.0
    is_playing: bool = False
    seek_requested: bool = False
    seek_time: float = 0.0
    playback_speed: float = 1.0

# -------------------------------------------------------------------
# EnhancedUNet Modified to Output a Single Channel (grayscale)
# -------------------------------------------------------------------
class EnhancedUNet(nn.Module):
    """U-Net for reversing fractal transformations"""
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

        # ***IMPORTANT***:
        # final_conv outputs 1 channel, ensuring grayscale decoding
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        """Helper function for double convolution layers"""
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

        # Output shape: [batch_size, 1, H, W]
        return self.final_conv(d1)

class ImagePairDataset(Dataset):
    """Dataset for image pairs (original and fractal)"""
    def __init__(self, original_dir, fractal_dir, image_pairs, transform=None):
        self.original_dir = original_dir
        self.fractal_dir = fractal_dir
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        orig_file, frac_file = self.image_pairs[idx]

        # Load images
        original_img = Image.open(os.path.join(self.original_dir, orig_file)).convert('L')
        fractal_img = Image.open(os.path.join(self.fractal_dir, frac_file)).convert('L')

        if self.transform:
            original_img = self.transform(original_img)
            fractal_img = self.transform(fractal_img)

        return fractal_img, original_img

class DecoderTest:
    """Testing framework for fractal-to-image decoding"""
    def __init__(self, model, device, resolution=512):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def process_single_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Test fractal encoding and decoding on a single image
        Returns: (original, fractal, decoded, PSNR, SSIM)
        """
        # Load and preprocess original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Invalid image or cannot open: {image_path}")

        # Resize to match the resolution
        original = cv2.resize(original, (self.resolution, self.resolution))

        # Create fractal version
        processor = FractalProcessor((0, 0), resolution=self.resolution)  # Dummy freq band
        fractal = processor.generate_fractal(np.mean(original) / 255.0)
        fractal = (fractal * 255).astype(np.uint8)

        # Decode fractal back to image
        fractal_tensor = self.transform(Image.fromarray(fractal)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            decoded_tensor = self.model(fractal_tensor)

            # decoded_tensor shape => [1, 1, 256, 256] if final_conv is out_channels=1
            decoded_np = decoded_tensor.squeeze(0).cpu().numpy()  # shape => [1, 256, 256]
            # If the network accidentally returns multiple channels, handle it:
            if decoded_np.shape[0] > 1:
                logger.warning(f"Model output has {decoded_np.shape[0]} channels. Using channel 0.")
            decoded_np = decoded_np[0]  # shape => [256, 256]
            # Convert from [0..1] to [0..255]
            decoded_np = (decoded_np * 255).astype(np.uint8)
            decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

        # Check shapes before computing PSNR/SSIM
        if original.shape != decoded_np.shape:
            raise ValueError(
                f"Shape mismatch: original={original.shape}, decoded={decoded_np.shape}. "
                "They must be identical for PSNR/SSIM."
            )

        # Calculate metrics
        psnr_val = psnr(original, decoded_np)
        ssim_val = ssim(original, decoded_np)

        return original, fractal, decoded_np, psnr_val, ssim_val

class VideoRecorder:
    """Handles recording of EEG visualization to video"""
    def __init__(self, resolution=512):
        self.resolution = resolution
        self.writer = None
        self.is_recording = False
        self.output_path = None

    def start_recording(self, output_path: str, fps: float = 30.0):
        """Start recording video"""
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.resolution * 2, self.resolution)
        )
        self.is_recording = True

    def add_frame(self, fractal_frame: np.ndarray, decoded_frame: np.ndarray):
        """Add a frame to the video"""
        if not self.is_recording:
            return

        fractal_frame = cv2.resize(fractal_frame, (self.resolution, self.resolution))
        decoded_frame = cv2.resize(decoded_frame, (self.resolution, self.resolution))

        combined = np.hstack([fractal_frame, decoded_frame])
        self.writer.write(combined)

    def stop_recording(self):
        """Stop recording and save video"""
        if self.writer:
            self.writer.release()
        self.is_recording = False
        self.writer = None

class ResultsLogger:
    """Logs and tracks decoder performance metrics"""
    def __init__(self, log_dir: str = "decoder_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []

    def log_result(self, original_path: str, psnr_val: float, ssim_val: float):
        """Log metrics for a single decode attempt"""
        self.metrics.append({
            'image': original_path,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'timestamp': time.time()
        })

    def save_metrics(self):
        """Save all metrics to file"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(self.log_dir, f"metrics_{timestamp}.txt")
        with open(log_path, 'w') as f:
            for m in self.metrics:
                f.write(f"{m['image']},{m['psnr']:.4f},{m['ssim']:.4f}\n")

    def get_summary(self) -> dict:
        """Get summary statistics of all metrics"""
        if not self.metrics:
            return {'psnr_avg': 0, 'ssim_avg': 0, 'psnr_std': 0, 'ssim_std': 0}

        psnr_vals = [m['psnr'] for m in self.metrics]
        ssim_vals = [m['ssim'] for m in self.metrics]

        return {
            'psnr_avg': np.mean(psnr_vals),
            'psnr_std': np.std(psnr_vals),
            'ssim_avg': np.mean(ssim_vals),
            'ssim_std': np.std(ssim_vals)
        }

# ------------------------------
# Main Application Class
# ------------------------------

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
        self.model.eval()  # Set model to evaluation mode
        self.last_update = time.time()

        # Additional components
        self.decoder_test = DecoderTest(self.model, self.device, self.resolution)
        self.video_recorder = VideoRecorder(self.resolution)
        self.results_logger = ResultsLogger()

        self.setup_gui()

    def setup_gui(self):
        """Set up the graphical user interface"""
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
        test_menu.add_command(label="Test Single Image", command=self.test_single_image)
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

        ttk.Label(control_frame, text="Time Position:").pack(pady=5)
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(control_frame, from_=0, to=100,
                                     variable=self.time_var, command=self.seek)
        self.time_slider.pack(fill=tk.X, padx=5, pady=5)

        for band in ['theta', 'alpha', 'beta', 'gamma']:
            frame = ttk.LabelFrame(control_frame, text=f"{band.title()} Band")
            frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(frame, text="Signal Strength:").pack()
            strength_var = tk.DoubleVar(value=50)
            strength_slider = ttk.Scale(frame, from_=0, to=100, variable=strength_var,
                                        command=lambda v, b=band: self.update_signal_strength(b, float(v)))
            strength_slider.pack(fill=tk.X)

            ttk.Label(frame, text="Coupling:").pack()
            coupling_var = tk.DoubleVar(value=0.3)
            coupling_slider = ttk.Scale(frame, from_=0, to=1, variable=coupling_var,
                                        command=lambda v, b=band: self.update_coupling(b, float(v)))
            coupling_slider.pack(fill=tk.X)

            ttk.Label(frame, text="Wave Speed:").pack()
            speed_var = tk.DoubleVar(value=1.0)
            speed_slider = ttk.Scale(frame, from_=0, to=5, variable=speed_var,
                                     command=lambda v, b=band: self.update_wave_speed(b, float(v)))
            speed_slider.pack(fill=tk.X)

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

    # ------------------------------
    # GUI Command Methods
    # ------------------------------

    def update_signal_strength(self, band: str, value: float):
        self.eeg.processors[band].signal_strength = value

    def update_coupling(self, band: str, value: float):
        self.eeg.processors[band].coupling_strength = value

    def update_wave_speed(self, band: str, value: float):
        self.eeg.processors[band].wave_speed = value

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
        try:
            channel_idx = self.eeg.raw.ch_names.index(self.channel_var.get())
            data = self.eeg.get_data(channel_idx, self.state.current_time)
            if data is None:
                return

            # Robust scaling
            data = np.asarray(data).flatten()
            data = data - np.median(data)
            mad = np.median(np.abs(data))
            data = data / (mad if mad > 1e-6 else 1.0)
            data = np.clip(data, -5, 5)
            data = (data + 5) / 10.0

            combined_fractal = np.zeros((self.resolution, self.resolution))
            for band, processor in self.eeg.processors.items():
                amplitude = np.abs(data).mean()
                fractal = processor.generate_fractal(amplitude)
                if band in ['theta', 'alpha', 'beta']:
                    next_band = {'theta': 'alpha', 'alpha': 'beta', 'beta': 'gamma'}[band]
                    coupling = processor.coupling_strength
                    next_fractal = self.eeg.processors[next_band].generate_fractal(amplitude)
                    fractal = fractal * (1 + coupling * next_fractal)
                combined_fractal += fractal

            combined_fractal = (combined_fractal - combined_fractal.min()) / (
                (combined_fractal.max() - combined_fractal.min()) + 1e-8
            )
            colored = cv2.applyColorMap(
                (combined_fractal * 255).astype(np.uint8), cv2.COLORMAP_JET
            )

            fractal_image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
            fractal_photo = ImageTk.PhotoImage(image=fractal_image)
            self.fractal_canvas.delete("all")
            self.fractal_canvas.create_image(0, 0, image=fractal_photo, anchor=tk.NW)
            self.fractal_canvas.image = fractal_photo

            fractal_tensor = self.decoder_test.transform(
                Image.fromarray((combined_fractal * 255).astype(np.uint8))
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                decoded_tensor = self.model(fractal_tensor)
                decoded_np = decoded_tensor.squeeze(0).cpu().numpy()
                # If there's more than 1 channel, just take channel 0
                if decoded_np.shape[0] > 1:
                    logger.warning(f"Visualization model output has {decoded_np.shape[0]} channels, using channel 0.")
                decoded_np = decoded_np[0]
                decoded_np = (decoded_np * 255).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (self.resolution, self.resolution))

            decoded_image = Image.fromarray(decoded_np)
            decoded_photo = ImageTk.PhotoImage(image=decoded_image)
            self.decoded_canvas.delete("all")
            self.decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            self.decoded_canvas.image = decoded_photo

            if self.video_recorder.is_recording:
                self.video_recorder.add_frame(colored, decoded_np)

        except Exception as e:
            logger.error(f"Error in visualization update: {e}")

    def test_single_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            results_window = tk.Toplevel(self.root)
            results_window.title("Decoder Test Results")

            original, fractal, decoded, psnr_val, ssim_val = self.decoder_test.process_single_image(filepath)

            images_frame = ttk.Frame(results_window)
            images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Original
            orig_frame = ttk.LabelFrame(images_frame, text="Original")
            orig_frame.pack(side=tk.LEFT, padx=5)
            orig_canvas = tk.Canvas(orig_frame, width=256, height=256)
            orig_canvas.pack()
            orig_photo = ImageTk.PhotoImage(Image.fromarray(original))
            orig_canvas.create_image(0, 0, image=orig_photo, anchor=tk.NW)
            orig_canvas.image = orig_photo

            # Fractal
            fractal_frame = ttk.LabelFrame(images_frame, text="Fractal")
            fractal_frame.pack(side=tk.LEFT, padx=5)
            fractal_canvas = tk.Canvas(fractal_frame, width=256, height=256)
            fractal_canvas.pack()
            fractal_photo = ImageTk.PhotoImage(Image.fromarray(fractal))
            fractal_canvas.create_image(0, 0, image=fractal_photo, anchor=tk.NW)
            fractal_canvas.image = fractal_photo

            # Decoded
            decoded_frame = ttk.LabelFrame(images_frame, text="Decoded")
            decoded_frame.pack(side=tk.LEFT, padx=5)
            decoded_canvas = tk.Canvas(decoded_frame, width=256, height=256)
            decoded_canvas.pack()
            decoded_photo = ImageTk.PhotoImage(Image.fromarray(decoded))
            decoded_canvas.create_image(0, 0, image=decoded_photo, anchor=tk.NW)
            decoded_canvas.image = decoded_photo

            # Metrics
            metrics_frame = ttk.Frame(results_window)
            metrics_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(metrics_frame, text=f"PSNR: {psnr_val:.2f} dB").pack(side=tk.LEFT, padx=10)
            ttk.Label(metrics_frame, text=f"SSIM: {ssim_val:.4f}").pack(side=tk.LEFT, padx=10)

            self.results_logger.log_result(filepath, psnr_val, ssim_val)

        except Exception as e:
            logger.error(f"Error in decoder test: {e}")
            messagebox.showerror("Error", f"Test failed: {str(e)}")

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
                original, fractal, decoded, psnr_val, ssim_val = self.decoder_test.process_single_image(path_)
                self.results_logger.log_result(path_, psnr_val, ssim_val)

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            self.results_logger.save_metrics()
            progress_window.destroy()
            messagebox.showinfo("Success", "Batch testing complete!")

        except Exception as e:
            logger.error(f"Error in batch testing: {e}")
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

        results_listbox = tk.Listbox(results_frame, yscrollcommand=scrollbar.set)
        for metric in self.results_logger.metrics:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric['timestamp']))
            results_listbox.insert(
                tk.END,
                f"{timestamp} | {os.path.basename(metric['image'])} | "
                f"PSNR: {metric['psnr']:.2f} dB | SSIM: {metric['ssim']:.4f}"
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

                combined_fractal = np.zeros((self.resolution, self.resolution))
                for processor in self.eeg.processors.values():
                    fractal = processor.generate_fractal(np.mean(image))
                    combined_fractal += fractal

                combined_fractal = (combined_fractal - combined_fractal.min()) / (
                    (combined_fractal.max() - combined_fractal.min()) + 1e-8
                )
                colored = cv2.applyColorMap(
                    (combined_fractal * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                output_path = os.path.join(output_dir, f"fractal_{filename}")
                cv2.imwrite(output_path, colored)

                progress = 100 * (i + 1) / len(image_files)
                progress_var.set(progress)
                status_var.set(f"Processed {i+1}/{len(image_files)} images")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", "Batch processing complete!")

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")

    def train_model(self):
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
            fractal_files = [f"fractal_{f}" for f in original_files]

            valid_pairs = []
            for orig, frac in zip(original_files, fractal_files):
                if os.path.exists(os.path.join(fractal_dir, frac)):
                    valid_pairs.append((orig, frac))

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

            num_epochs = 200
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

                status_var.set(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("Success", f"Training complete!\nBest validation loss: {best_val_loss:.4f}")

        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")

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
            logger.error(f"Error starting recording: {e}")
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        if not self.video_recorder.is_recording:
            messagebox.showwarning("Recording", "Not currently recording!")
            return
        try:
            self.video_recorder.stop_recording()
            messagebox.showinfo("Recording", "Recording stopped and saved.")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
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
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def save_model(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if filepath:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                }, filepath)
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

def main():
    root = tk.Tk()
    app = FractalEEGApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()
