

# Eldians: Vision Assistance for the Blind

VISIONVISTA is a Generative AI system designed to assist blind individuals by providing information about their surroundings. The system uses object detection and depth estimation models to detect objects in images and estimate their distances. The Streamlit interface facilitates user interaction through both image uploads and speech commands.

## Getting Started

Follow these instructions to set up and run the VISIONVISTA system on your local machine.

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- [Hugging Face Transformers](https://github.com/huggingface/transformers) library
- [Streamlit](https://streamlit.io/) library
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) library
- [pyttsx3](https://pypi.org/project/pyttsx3/) library
- [Matplotlib](https://matplotlib.org/) library
- [NumPy](https://numpy.org/) library
- [Pillow (PIL)](https://pillow.readthedocs.io/) library

Install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

### Usage

1. Clone the repository:

```bash
git clone https://github.com/Lourarhi-Yahya/Hackaton2023_GenAI
cd Hackaton2023_GenAI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run interface.py
```

4. Follow the on-screen instructions to upload an image and initiate a conversation.

## System Architecture

- `COMPUTER_Vision.py`: Contains the computer vision functions for object detection and depth estimation.
- `interface.py`: Implements the Streamlit interface and integrates the computer vision system with speech interaction.

## Speech Interaction

The system uses the Google Speech Recognition service to convert spoken words to text. Make sure you have a reliable internet connection for accurate speech recognition.

## Quality and Contributions

We welcome contributions to enhance VisionVista. If you encounter issues or have suggestions, feel free to open an [issue](https://github.com/Lourarhi-Yahya/Hackaton2023_GenAI/issues) or submit a pull request.

Please follow these guidelines for contributing:

- Clearly describe the problem or enhancement.
- Provide sample code or steps to reproduce the issue.
- Ensure the code follows best practices and is well-documented.

## License

This project is licensed under the [MIT License](LICENSE.md).
