# VLM-Based Safety Violation Detection

A vision-language-model-based prototype for assessing construction-site safety compliance from video footage. It uses Google's Gemini vision model to assess sampled video frames without YOLO or a separate object-detection model.

## Features

- Upload construction-site videos for safety assessment
- Configurable PPE safety rules for helmets and high-visibility vests
- Video frame extraction and configurable sampling intervals
- Timestamped **VIOLATION**, **COMPLIANT**, and **UNCERTAIN** assessments
- Evidence-based explanations and recommended actions
- Streamlit interface for reviewing assessment results
- JSON report export

## Architecture

Video upload → OpenCV frame extraction and sampling → Gemini vision model → Structured assessment → Timestamped report and human review

OpenCV is used for video processing, not object detection.

## Requirements

- Python 3.10 or newer
- Google Gemini API key
- Internet connection

## Setup on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/musthafa145/VLM_Based_Violation_Detection.git
cd VLM_Based_Violation_Detection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Replace the placeholder with your own API key. Use a Gemini model name that is available to your account and supported by your application.

**Security:** Never commit `.env` or publish your API key. Keep it excluded through `.gitignore`.

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Limitations

This is a prototype, not a fully validated safety-monitoring system. Model assessments may be incorrect or inconsistent. Frame sampling can miss short-lived violations, and results do not establish reliable identity or continuous tracking of individual workers. Human review is required before taking enforcement or other consequential actions.

The system's accuracy and reliability have not been established through a systematic evaluation.

## Technology Stack

- Python
- Google Gemini vision model
- Google Gen AI Python SDK
- Streamlit
- OpenCV
- Pillow
- python-dotenv
