# MLLM

# Intent and Slot Predictor

## Overview
This project implements an **Intent and Slot Predictor** using **FastAPI** as the backend and a **chatbot-style frontend**. The model is built using **XLM-RoBERTa**, a transformer-based model for multi-lingual text classification and slot tagging tasks.

The model is designed to predict:
- **Intent**: The main purpose of the sentence (e.g., book a flight, ask for the weather).
- **Slots**: Key entities or pieces of information mentioned in the sentence (e.g., "New York" as a destination).

This app can be used to predict intents and slots from a sentence input in a conversational style interface.

## Features
- Chatbot interface with a dark theme.
- Predicts intent and slot labels based on the input sentence.
- Built with **FastAPI** for the backend and **HTML/CSS/JS** for the frontend.
- Pre-trained model using **XLM-RoBERTa**.

## Setup

### Requirements
- Python 3.8+
- Git
- [Google Chrome](https://www.google.com/chrome/) (for optimal experience)

### Installing Dependencies
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/MLLM.git
   cd MLLM
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   pip install -r requirements.txt
   ```

3. Install **Git LFS** to handle large files (like the model):
   - Download and install **Git LFS** from [here](https://git-lfs.github.com/).
   - Initialize Git LFS:
     ```bash
     git lfs install
     ```

### Model File (3GB)

Since the model file is too large to be uploaded directly to GitHub, we have stored it on **Google Drive**. You can download it using the link below.

#### Download the Pre-trained Model:
- [Download the `best_model.pt` from Google Drive](https://drive.google.com/file/d/your-google-drive-id/view?usp=sharing)

### Place the Model File
After downloading the model file, place it in the root directory of this project (the same location as `app.py`). Rename it to `best_model.pt`.

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

2. Open your browser and go to:
   ```
   http://127.0.0.1:8000
   ```

You should now see the **Intent and Slot Predictor** chatbot interface.

### Example Usage

1. Type a sentence in the chat box (e.g., "Book a flight to New York").
2. The system will predict the intent and extract the slots (e.g., "transport_ticket", "New York").

### Frontend Details

The frontend is designed with HTML, CSS, and JavaScript to provide a sleek, modern, and user-friendly chatbot interface. It uses **JavaScript** to send input data to the FastAPI backend and display predictions in a conversational manner.

## Technologies Used
- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Model**: XLM-RoBERTa for Intent Classification and Slot Tagging
- **Git LFS**: For handling large files
- **Pytorch**: For running the model

## Project Structure
```
MLLM/
├── app.py                    # FastAPI backend code
├── static/                    # Static files (CSS, JS)
│   ├── styles.css
│   ├── script.js
├── templates/                 # HTML file for UI
│   └── index.html
├── best_model.pt              # Your 3GB model (tracked with Git LFS)
├── requirements.txt           # Backend dependencies
└── README.md                  # Project information
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Thanks to **XLM-RoBERTa** for providing a powerful pre-trained model for multi-lingual tasks.
- Thanks to **GitHub** and **Google Drive** for helping us store and share large files.

---

Feel free to open an issue if you face any problems or have questions about the setup.
```

### Steps to follow:

1. **Replace the Google Drive link**: You need to upload your model file (`best_model.pt`) to **Google Drive** and replace the link in the README above (`https://drive.google.com/file/d/your-google-drive-id/view?usp=sharing`). Make sure the model file is set to "Anyone with the link can view."

2. **Place the Model File**: After downloading the model from Google Drive, place it in your local project folder, with the file name `best_model.pt`.

3. **Add Your Repository Link**: Replace `your-username` in the Git clone URL with your actual GitHub username.

### Final Step: Push the Changes to GitHub

Once you've updated the `README.md` file, you can commit and push it to GitHub:

```bash
git add README.md
git commit -m "Add instructions for model download and project setup"
git push origin main
```

This will ensure that your users can easily set up and use the model in the app!
