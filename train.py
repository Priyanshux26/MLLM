import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import XLMRobertaTokenizerFast
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn
from transformers import XLMRobertaModel
from torch.optim.lr_scheduler import StepLR


# Load the dataset
df = pd.read_csv("intentslotfinal.csv")  # Replace with your dataset path

# Parse the slot column to extract tokens and BIO labels
def parse_slot_column(slot_col):
    tokens, slot_labels = [], []
    for pair in slot_col.split(','):
        token, label = pair.split(':')
        tokens.append(token)
        slot_labels.append(label)
    return tokens, slot_labels

# Define custom dataset
class JointIntentSlotDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = []
        for _, row in df.iterrows():
            tokens, slot_labels = parse_slot_column(row['slot'])
            intent = row['intent']

            # Tokenize and align slot labels
            encodings = tokenizer(tokens, truncation=True, padding="max_length", max_length=max_length, is_split_into_words=True)
            label_ids = [-100] * len(encodings['input_ids'])
            word_ids = encodings.word_ids()
            label_index = 0
            for i, word_id in enumerate(word_ids):
                if word_id is not None and label_index < len(slot_labels):
                    label_ids[i] = tokenizer.slot_label_to_id[slot_labels[label_index]]
                    label_index += 1

            self.data.append({
                "input_ids": encodings['input_ids'],
                "attention_mask": encodings['attention_mask'],
                "intent_label": tokenizer.intent_label_to_id[intent],
                "slot_labels": label_ids
            })

    def __len__(self):  # Corrected method for len()
        return len(self.data)

    def __getitem__(self, idx):  # Corrected method for item access
        return self.data[idx]

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    intent_labels = torch.tensor([item['intent_label'] for item in batch], dtype=torch.long)
    slot_labels = torch.tensor([item['slot_labels'] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "intent_label": intent_labels, "slot_labels": slot_labels}

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
        self._init_weights()

    def _init_weights(self):
        for module in [self.intent_classifier, self.slot_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Take [CLS] token

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = CrossEntropyLoss(weight=intent_class_weights)
            slot_loss_fn = CrossEntropyLoss(ignore_index=-100, weight=slot_class_weights)

            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
            loss = 0.5 * intent_loss + 0.5 * slot_loss  # Adjust weights as needed

        return {"loss": loss, "intent_logits": intent_logits, "slot_logits": slot_logits}

# Define the tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

# Create mappings
all_slot_labels = set()
for slot_string in df['slot']:
    for pair in slot_string.split(','):
        _, label = pair.split(':')
        all_slot_labels.add(label)

tokenizer.slot_label_to_id = {label: idx for idx, label in enumerate(all_slot_labels.union(["O"]))}
tokenizer.intent_label_to_id = {label: idx for idx, label in enumerate(df['intent'].unique())}

# Create dataset and split into train/test
# Define collate function
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    intent_labels = torch.tensor([item['intent_label'] for item in batch], dtype=torch.long)
    slot_labels = torch.tensor([item['slot_labels'] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "intent_label": intent_labels, "slot_labels": slot_labels}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_intent_labels = len(tokenizer.intent_label_to_id)
num_slot_labels = len(tokenizer.slot_label_to_id)
model = JointIntentSlotModel("xlm-roberta-base", num_intent_labels, num_slot_labels).to(device)

# Create dataset and split into train/test
data = JointIntentSlotDataset(df, tokenizer)
test_size = 500
train_size = len(data) - test_size
train_data, test_data = random_split(data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn)


# Define collate function

# Optimizer and Scheduler
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=2, gamma=0.8)

# Define class weights
intent_class_weights = torch.ones(len(tokenizer.intent_label_to_id)).to(device)
slot_class_weights = torch.ones(len(tokenizer.slot_label_to_id)).to(device)
slot_class_weights[tokenizer.slot_label_to_id["O"]] = 0.5  # Adjust as needed

# Training loop
best_intent_accuracy = 0
for epoch in range(10):
    print(f"Epoch {epoch + 1} starting...")
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", unit="batch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels = batch['intent_label'].to(device)
        slot_labels = batch['slot_labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, intent_labels=intent_labels, slot_labels=slot_labels)
        loss = outputs["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")
    scheduler.step()

    # Evaluate on test set
    model.eval()
    total_intent_accuracy = 0
    total_slot_accuracy = 0
    total_slots = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_label'].to(device)
            slot_labels = batch['slot_labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            intent_logits = outputs["intent_logits"]
            slot_logits = outputs["slot_logits"]

            # Intent accuracy
            intent_preds = intent_logits.argmax(dim=-1)
            total_intent_accuracy += accuracy_score(intent_labels.cpu().numpy(), intent_preds.cpu().numpy())

            # Slot accuracy
            slot_preds = slot_logits.argmax(dim=-1)
            slot_mask = slot_labels != -100
            slot_flattened_preds = slot_preds[slot_mask]
            slot_flattened_labels = slot_labels[slot_mask]
            total_slot_accuracy += accuracy_score(slot_flattened_labels.cpu().numpy(), slot_flattened_preds.cpu().numpy())
            total_slots += 1

    avg_intent_accuracy = total_intent_accuracy / len(test_loader)
    avg_slot_accuracy = total_slot_accuracy / total_slots
    print(f"Test Results: Intent Accuracy = {avg_intent_accuracy:.4f}, Slot Accuracy = {avg_slot_accuracy:.4f}")

    # Save the best model along with optimizer state
    if avg_intent_accuracy > best_intent_accuracy:
        best_intent_accuracy = avg_intent_accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }, "best_model.pt")