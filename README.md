# Trail Cam Classifier

Automatically extract the best frames from wildlife trail camera footage and classify them using AI.

This project uses computer vision to find the sharpest, most interesting frames in a video (filtering out swaying trees or empty shots) and can optionally send a 2x2 grid of those frames to Google Gemini for species identification.

## 🚀 How to Use

### 1. Quick Start (Extraction Only)
Run the script on your video folder (default: `video-samples/`):
```bash
uv run main.py
```
This will create an `extracted-frames/` directory with the best individual frames and a JSON index for each video.

### 2. Extract & Classify (AI Mode)
To automatically identify animals using Google Gemini:
1.  Add your API key to a `.env` file: `GEMINI_API_KEY=your_key_here`
2.  Run with the `--classify` and `--tile` flags:
```bash
uv run main.py --tile --classify
```
The `--tile` flag creates a 2x2 "hero tile" (4 frames in one image), which allows the AI to see multiple angles of the same event while saving on API tokens.

---

## 🛠 Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

1.  **Install `uv`**:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows
    powershell -c "ir https://astral.sh/uv/install.ps1 | iex"
    ```
2.  **Clone & Run**:
    ```bash
    git clone https://github.com/your-username/trail-cam-classifier.git
    cd trail-cam-classifier
    uv run main.py
    ```

---

## ✨ Features

-   **Intelligent Frame Selection**: Scores frames based on sharpness (Laplacian variance), motion (MOG2 background subtraction), and ideal brightness.
-   **Temporal Deduplication**: Ensures you don't get 10 frames of the same second; it spaces out selections and uses perceptual hashing (aHash) to avoid nearly identical shots.
-   **VLM Grid Tiling**: Combines the top 4 candidates into a single 2x2 grid for efficient Vision Language Model processing.
-   **Gemini 2.0 Integration**: Uses Google's latest models to provide structured JSON data about what animals are present and how certain the AI is.

## ⚙️ Advanced Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--input, -i` | Path to video file or directory | `video-samples` |
| `--per-video, -k` | Number of frames to save per video | `8` |
| `--tile` | Generate a 2x2 grid of the best frames | `False` |
| `--classify` | Send the hero tile to Gemini for AI ID | `False` |
| `--sample-every` | Seconds between sampled frames | `0.5` |
| `--weights` | Scoring weights (sharpness,motion,brightness) | `0.45,0.45,0.1` |

Full list of options available via `uv run main.py --help`.
