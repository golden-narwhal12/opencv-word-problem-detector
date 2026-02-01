# Word Problem Detector

Real-time computer vision system that detects word problems on paper and captures high-quality images.

## What It Does

Point your camera (webcam or IPCamera) at a piece of paper with a word problem on it. The program detects when it sees text that looks like a word problem, then captures a sharp image when you press spacebar. Uses burst capture to get the sharpest possible image even if your hand shakes.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Webcam:**
```bash
python3 word_problem_demo.py 0
```

**IP Camera:**
```bash
python3 word_problem_demo.py rtsp://admin:password@192.168.1.100
```

**Auto-capture mode (captures automatically after 2 seconds of stable detection):**
```bash
python3 word_problem_demo.py 0 --auto-capture
```

**Controls:**
- SPACEBAR - Start scanning / Capture image
- Q - Quit

## How It Works

**Detection:**
The system looks for text by checking brightness, contrast, and edge patterns. It groups text into lines and makes sure the text isn't cut off at the edges. Once it sees enough text that looks like a complete problem (60% confidence), it tells you it's ready to capture.

**Capture:**
When you hit spacebar, it captures 120 frames over 3.5 seconds and picks the sharpest one. This handles hand shake and motion blur really well. The best frame gets saved to the captures folder.

**Performance:**
Processes frames at 640px width for speed. Spatial hashing optimization makes neighbor detection ~20x faster. Runs detection every other frame to keep the video feed smooth.

## Options

**Auto-capture mode:** Add --auto-capture flag to automatically capture after 2 seconds of stable detection

**Voice guidance:** Edit word_problem_demo.py line 73 and set `self.enable_voice_guidance = True`. The program will tell you if text is cut off or the camera is too close. Works on macOS without extra dependencies. Other platforms need: `pip install pyttsx3` (included in requirements.txt)

## Troubleshooting

**Camera not found:** Try different device numbers (0, 1, 2)

**Low confidence:** Make sure text isn't cut off at edges and lighting is decent

**Blurry images:** Hold still during the 3.5 second capture

## Output

Images saved to captures/word_problem_YYYYMMDD_HHMMSS.png

## Requirements

- Python 3.7+
- OpenCV (computer vision)
- NumPy (array processing)
- pyttsx3 (optional, for voice guidance on non-macOS)
