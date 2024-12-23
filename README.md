# EDF-EEG-FractalVisualizer

EDIT: 
Yes, the code makes similar structures out of the temp channel etc too. So the code produces a  lot of the fractal likeness. But the fractals are more complex with eeg data.

Claude Pointed out: 

These create fractal-like patterns for any input, but:

With temperature: Shows basic wave dynamics
With EEG: Reveals complex frequency interactions


Key Differences:


Temperature channel shows smoother gradients
Activity channel shows more structured boundaries
EEG shows rapid pattern formation/dissolution

![Visualizer](./visualizer.png)


Real-time EEG visualization tool that displays brain wave frequency bands as dynamic wave patterns.

## Installation

Install dependencies:
```bash
pip install numpy pillow mne scipy opencv-python etc. 
```

## Usage

1. Run the script:
```bash
python eeg_visualizer.py
```

2. Click "Load EEG" to select an EEF file. Wait a while until channel selection becomes available. 
3. Select a channel from the dropdown
4. Use the Play/Pause button to control visualization
5. Adjust sliders to control wave behavior for each frequency band
