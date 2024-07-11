# Emotion Detection in Speech Across Languages

## Introduction
Our goal is to determine if there are features of spoken language that can be
used to determine whether a statement is spoken sadly as opposed to happily
or angrily, regardless of what language the speaker is speaking and whether
the listener understands the words of that language or not.

## Installation
To set up your virtual environment and install the required dependencies, run the following commands:

```bash
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate the virtual environment (Linux/Mac)
venv\Scripts\activate  # Activate the virtual environment (Windows)
pip install -r requirements.txt  # Install dependencies
```
and afterwards, 
## Usage
This project includes a script named `predict_audio_file.py` that you can use to predict the emotion in an audio file. To use this script, follow these steps:

1. Ensure that you have activated the virtual environment and that all dependencies are installed.
2. Prepare your audio file and name it according to the required format: `{first 3 letters of language}_{M for male or F for female}_{additional identifiers}.wav`.
3. Run the script from the command line by passing the file name as an argument:

```bash
python predict_audio_file.py <path_to_your_audio_file>
```

## Model Details
- **Logistic Regression Model**: Achieved over % accuracy in predicting sadness in audio clips.
- **XGBoost Model**: Performed slightly better than the logistic regression model. getting % accuracy.

## Contributors
This project was made possible by the hard work and dedication of the following individuals:
- Sara Page Podolsky
- Danny Kaplunski



