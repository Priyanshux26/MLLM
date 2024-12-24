from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
import torch.nn as nn
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Load dataset
try:
    df = pd.read_csv('intentslotfinal.csv')
except FileNotFoundError:
    raise Exception("The dataset file 'intentslotfinal.csv' was not found. Please ensure the file is in the correct location.")

# Define the model architecture
class JointIntentSlotModel(nn.Module):
    def __init__(self, model_name, num_intent_labels, num_slot_labels):
        super(JointIntentSlotModel, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_intent_labels)
        )
        self.slot_classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_slot_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Hidden states for all tokens
        cls_output = sequence_output[:, 0, :]  # Take the [CLS] token for intent prediction

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)  # Slot prediction for all tokens

        return {"intent_logits": intent_logits, "slot_logits": slot_logits}

# Initialize the FastAPI app
app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Input data model
class InputData(BaseModel):
    sentence: str

# Load the tokenizer and model
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
intent_label_to_id = {label: idx for idx, label in enumerate(df['intent'].unique())}
id_to_intent_label = {idx: label for label, idx in intent_label_to_id.items()}

all_slot_labels = set()
for slot_string in df['slot']:
    for pair in slot_string.split(','):
        if ':' in pair:
            _, label = pair.split(':')
            all_slot_labels.add(label)

slot_label_to_id = {label: idx for idx, label in enumerate(all_slot_labels.union(["O"]))}
id_to_slot_label = {idx: label for label, idx in slot_label_to_id.items()}

NUM_INTENT_LABELS = len(intent_label_to_id)
NUM_SLOT_LABELS = len(slot_label_to_id)

MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = JointIntentSlotModel("xlm-roberta-base", NUM_INTENT_LABELS, NUM_SLOT_LABELS)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Return the HTML page to the client
    html_content = Path("static/index.html").read_text()
    return HTMLResponse(content=html_content)

@app.post("/predict")
def predict(data: InputData):
    try:
        # Tokenize the input sentence
        encodings = tokenizer(
            data.sentence, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            intent_logits = outputs["intent_logits"]
            slot_logits = outputs["slot_logits"]

        # Decode the intent prediction
        intent_pred = torch.argmax(intent_logits, dim=-1).item()
        intent_label = id_to_intent_label[intent_pred]

        # Decode the slot predictions
        slot_preds = torch.argmax(slot_logits, dim=-1).squeeze().tolist()
        tokenized_words = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        # Get slot labels for each token in the sentence
        slot_labels = [
            id_to_slot_label.get(slot_preds[i], "O") for i in range(len(tokenized_words))
        ]

        # Filter out special tokens (e.g., [CLS], [SEP])
        filtered_slot_labels = [
            slot_labels[i] for i in range(len(tokenized_words)) if tokenized_words[i] not in tokenizer.all_special_tokens
        ]

        return {"intent": intent_label, "slots": filtered_slot_labels}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
