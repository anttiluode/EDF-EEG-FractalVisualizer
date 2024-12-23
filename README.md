# EEG Visualizer and Fractal Decoder

![Visualizer Screenshot](visualizer.png)

This project consists of two complementary applications:

## 1. EEG Visualizer (Fractal Maker)

A Python application that generates fractal-like visualizations from EEG data or images. It processes different frequency bands and creates dynamic, colorful visualizations that represent the underlying patterns in the data.

### Features
- Load and visualize EEG data (.edf files)
- Process input images into fractal-like patterns
- Real-time visualization with adjustable parameters
- Frequency band controls (Theta, Alpha, Beta, Gamma)
- Adjustable wave speed, decay rate, and coupling strength
- Save generated visualizations

### Dependencies
```
numpy
tkinter
pillow
mne
scipy
opencv-python
pywt
```

### Usage
1. Run the EEG Visualizer:
```bash
python eeg_visualizer.py
```
2. Either:
   - Load an EEG file (.edf) and select a channel
   - Process an input image
3. Adjust the band controls to modify the visualization
4. Save the generated fractal image

## 2. Fractal Decoder

A deep learning-based application that attempts to reverse the fractal transformation process, reconstructing original images from their fractal representations.

### Features
- Train on pairs of original and fractalized images
- Enhanced U-Net architecture with frequency awareness
- Real-time training visualization
- Save and load trained models
- Decode new fractal images
- Multi-scale reconstruction

### Dependencies
```
torch
torchvision
pillow
scikit-learn
numpy
tkinter
```

### Usage
1. Run the Fractal Decoder:
```bash
python fractal_decoder.py
```
2. Load training data:
   - Add original images
   - Add corresponding fractalized images
3. Train the model
4. Use "Decode Image" to reconstruct original images from fractals

### Training Tips
- Start with a small learning rate (0.001)
- Use paired images (original and their fractal versions)
- Train for at least 50 epochs
- Save the model periodically

## Note

This project was developed with assistance from Claude (Anthropic) and demonstrates an experimental approach to visualizing and processing EEG data and images using fractal-like transformations.

## License

MIT License

## Acknowledgments

Special thanks to Claude (Anthropic) for assistance in developing this project.
