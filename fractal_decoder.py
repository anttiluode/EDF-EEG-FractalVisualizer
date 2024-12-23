import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.dec4 = self._double_conv(1024, 512)  # 512 + 512 input
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)   # 256 + 256 input
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)   # 128 + 128 input
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)    # 64 + 64 input
        
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
        # Main encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Frequency analysis path
        f1 = self.freq_conv1(x)
        f2 = self.freq_conv2(self.freq_pool(f1))
        
        # Decoder path with skip connections
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
        
        # Final output
        out = self.final_conv(d1)
        
        # Multi-scale outputs for loss computation
        s1 = F.interpolate(out, scale_factor=0.25)
        s2 = F.interpolate(out, scale_factor=0.5)
        
        return out, [s1, s2, out]

class EnhancedLoss(nn.Module):
    def __init__(self):
        super(EnhancedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, multi_scale_outputs=None):
        # Main reconstruction loss
        main_loss = 0.5 * (self.l1_loss(pred, target) + self.mse_loss(pred, target))
        
        # Multi-scale loss
        if multi_scale_outputs is not None:
            scale_loss = 0
            for output in multi_scale_outputs:
                scaled_target = F.interpolate(target, size=output.shape[2:], mode='bilinear')
                scale_loss += 0.1 * (self.l1_loss(output, scaled_target) + self.mse_loss(output, scaled_target))
            return main_loss + scale_loss
            
        return main_loss

class ImageDataset(Dataset):
    def __init__(self, original_paths, fractal_paths, transform=None):
        self.original_paths = original_paths
        self.fractal_paths = fractal_paths
        self.transform = transform

    def __len__(self):
        return len(self.original_paths)

    def __getitem__(self, idx):
        try:
            original = Image.open(self.original_paths[idx]).convert('L')
            fractal = Image.open(self.fractal_paths[idx]).convert('L')

            if self.transform:
                original = self.transform(original)
                fractal = self.transform(fractal)

            return fractal, original
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            blank = torch.zeros(1, 256, 256)
            return blank, blank

class FractalReversalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractal Reversal Training and Decoding")
        
        # Initialize model and training components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedUNet().to(self.device)
        self.criterion = EnhancedLoss().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Dataset paths
        self.original_paths = []
        self.fractal_paths = []
        
        # Transform for image processing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Training state
        self.current_epoch = 0
        self.max_epochs = 50
        self.best_loss = float('inf')
        
        # Setup GUI
        self.setup_gui()
        
        logger.info(f"Using device: {self.device}")

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(buttons_frame, text="Add Original Images", command=self.load_original_images).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Add Fractalized Images", command=self.load_fractal_images).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="Train Model", command=self.train_model).grid(row=0, column=2, padx=5)
        ttk.Button(buttons_frame, text="Stop Training", command=self.stop_training).grid(row=0, column=3, padx=5)
        ttk.Button(buttons_frame, text="Save Model", command=self.save_model).grid(row=0, column=4, padx=5)
        ttk.Button(buttons_frame, text="Load Model", command=self.load_model).grid(row=0, column=5, padx=5)
        ttk.Button(buttons_frame, text="Decode Image", command=self.decode_image).grid(row=0, column=6, padx=5)

        # Training parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, padx=5)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=2, padx=5)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=3, padx=5)

        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=1, column=0)

        # Image display frame
        display_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="5")
        display_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.canvas_input = tk.Canvas(display_frame, width=256, height=256, bg='white')
        self.canvas_input.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(display_frame, text="Input Image").grid(row=1, column=0)
        
        self.canvas_output = tk.Canvas(display_frame, width=256, height=256, bg='white')
        self.canvas_output.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(display_frame, text="Output Image").grid(row=1, column=1)

    def load_original_images(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if paths:
            self.original_paths.extend(paths)
            self.status_var.set(f"Loaded {len(paths)} original images. Total: {len(self.original_paths)}")

    def load_fractal_images(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if paths:
            self.fractal_paths.extend(paths)
            self.status_var.set(f"Loaded {len(paths)} fractal images. Total: {len(self.fractal_paths)}")

    def train_model(self):
        if not self.original_paths or not self.fractal_paths:
            messagebox.showerror("Error", "Please load both original and fractal images first.")
            return
            
        if len(self.original_paths) != len(self.fractal_paths):
            messagebox.showerror("Error", "Number of original and fractal images must match.")
            return

        try:
            # Update learning rate
            new_lr = float(self.lr_var.get())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # Create dataset and dataloader
            dataset = ImageDataset(self.original_paths, self.fractal_paths, transform=self.transform)
            self.dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

            self.max_epochs = int(self.epochs_var.get())
            self.current_epoch = 0
            self.best_loss = float('inf')
            
            # Start training loop
            self.train_epoch()
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")

    def train_epoch(self):
        if self.current_epoch >= self.max_epochs:
            self.status_var.set("Training completed!")
            return

        try:
            self.model.train()
            epoch_loss = 0
            batch_count = len(self.dataloader)

            for batch_idx, (fractal, original) in enumerate(self.dataloader):
                fractal, original = fractal.to(self.device), original.to(self.device)

                self.optimizer.zero_grad()
                output, multi_scale = self.model(fractal)
                loss = self.criterion(output, original, multi_scale)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                progress = (batch_idx + 1) / batch_count * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Epoch {self.current_epoch + 1}/{self.max_epochs}, Batch {batch_idx + 1}/{batch_count}")

            avg_loss = epoch_loss / batch_count
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint('best_model.pth')

            self.current_epoch += 1
            self.root.after(100, self.train_epoch)

        except Exception as e:
            logger.error(f"Error during training: {e}")
            messagebox.showerror("Error", f"Training error: {str(e)}")

    def stop_training(self):
        self.current_epoch = self.max_epochs
        self.status_var.set("Training stopped by user")

    def save_checkpoint(self, filepath):
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'best_loss': self.best_loss
            }, filepath)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            messagebox.showerror("Error", f"Failed to save checkpoint: {str(e)}")

    def save_model(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pth",
                filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
            )
            if file_path:
                self.save_checkpoint(file_path)
                messagebox.showinfo("Success", "Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
            )
            if file_path:
                checkpoint = torch.load(file_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_epoch = checkpoint['epoch']
                self.best_loss = checkpoint['best_loss']
                messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def decode_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
            )
            if not file_path:
                return

            # Load and preprocess input image
            input_image = Image.open(file_path).convert('L')
            original_size = input_image.size
            input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

            # Process through model
            self.model.eval()
            with torch.no_grad():
                output, _ = self.model(input_tensor)
                reconstructed = output.squeeze(0).cpu()

            # Convert to PIL Image and resize to original dimensions
            reconstructed_image = transforms.ToPILImage()(reconstructed)
            reconstructed_image = reconstructed_image.resize(original_size, Image.BICUBIC)

            # Display input image
            input_display = input_image.resize((256, 256))
            input_photo = ImageTk.PhotoImage(input_display)
            self.canvas_input.create_image(128, 128, image=input_photo)
            self.canvas_input.image = input_photo

            # Display output image
            output_display = reconstructed_image.resize((256, 256))
            output_photo = ImageTk.PhotoImage(output_display)
            self.canvas_output.create_image(128, 128, image=output_photo)
            self.canvas_output.image = output_photo

        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            messagebox.showerror("Error", f"Failed to decode image: {str(e)}")

def main():
    root = tk.Tk()
    app = FractalReversalApp(root)
    root.geometry("800x900")
    root.mainloop()

if __name__ == "__main__":
    main()