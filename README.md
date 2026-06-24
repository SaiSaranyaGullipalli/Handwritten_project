# Smart Handwriting Recognition

Smart Handwriting Recognition is a Python desktop application that extracts handwritten text from images, improves the OCR output, and gives the user a few post-processing tools inside a simple Tkinter interface.

The current application entry point is `main.py`.

## Features

- Handwritten OCR using Hugging Face Transformers and Microsoft's TrOCR handwritten model
- Image preprocessing with OpenCV to improve recognition quality
- Automatic line detection for multi-line handwritten notes
- Grammar and spelling cleanup after OCR
- Synonym lookup using NLTK WordNet
- Offline text-to-speech playback with `pyttsx3`
- Local image preview and recognized text editing in the GUI
- Text export from the app

## How It Works

The OCR pipeline combines classical image processing with transformer-based recognition:

1. Extracts ink while suppressing the paper background
2. Denoises and enhances contrast
3. Builds a binary handwriting mask
4. Detects the text area and separates lines
5. Normalizes each line image for OCR
6. Runs TrOCR to decode handwritten text
7. Filters weak candidates and keeps the strongest results
8. Applies cleanup, spelling correction, and grammar rules

Additional algorithm notes are documented in `algorithm.txt`.

## Tech Stack

- Python
- Tkinter
- OpenCV
- NumPy
- Pillow
- PyTorch
- Transformers
- Microsoft TrOCR
- NLTK / WordNet
- PySpellChecker
- pyttsx3

Additional tool notes are documented in `tools.txt`.

## Project Structure

```text
.
|-- main.py                 # Main desktop application
|-- requirements.txt        # Python dependencies
|-- algorithm.txt           # Summary of OCR and preprocessing algorithms
|-- tools.txt               # Summary of tools and libraries used
|-- models/
|   `-- trocr_cache/        # Cached TrOCR model files
|-- output_processed*.png   # Sample/generated processed image outputs
|-- main1.1.py, main4.py... # Earlier development versions / experiments
```

## Installation

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Model Requirement

`main.py` is configured to load TrOCR in offline mode using local cached files:

- cache directory: `models/trocr_cache`
- model loading uses `local_files_only=True`

That means the app will only start successfully if the TrOCR model files already exist in the cache directory.

This repository already contains a populated `models/trocr_cache` folder, so the app should run offline in this checkout. If that folder is removed, the current code will not automatically download the model.

## Running the App

```powershell
python main.py
```

## Using the App

1. Launch the application.
2. Click `Get Started`.
3. Use `Upload` to select an image file.
4. Click `Convert` to recognize the handwriting.
5. Optionally use:
   - `Grammar & Spell` to clean the text
   - `Synonyms` to replace a selected word
   - `Phonate` to read the text aloud
   - `To PDF` to export the recognized text
6. Use `Delete Image` to clear the current image and text.

Supported image formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`

## Important Notes

- GPU is optional. The app uses CUDA automatically when available, otherwise it runs on CPU.
- The TrOCR model is large, so startup and inference can be slower on CPU-only systems.
- NLTK may try to download `wordnet` and `omw-1.4` on first run if they are missing locally.
- The current `To PDF` button in `main.py` exports plain text through a save dialog. The button label does not yet match the implementation.
- A Tesseract installer is present in the repository, but the current `main.py` flow uses TrOCR, not Tesseract OCR.

## Known Limitations

- Recognition quality depends heavily on image clarity, lighting, and handwriting style.
- Very noisy, skewed, or low-contrast images may still produce weak OCR output.
- The repository contains multiple historical script versions, so `main.py` should be treated as the active app unless you intentionally want to inspect earlier iterations.

## Future Improvement Ideas

- Match the export button behavior to its label by generating real PDFs
- Add batch image processing
- Improve paragraph reconstruction for long handwritten pages
- Add packaging for easier desktop distribution

