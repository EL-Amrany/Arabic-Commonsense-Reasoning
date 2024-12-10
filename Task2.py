import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set a random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define the custom dataset
class ReasonPredictionDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-large-arabic")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve the three sentences and the label
        sentence1 = self.dataframe.iloc[idx]['sentence1']
        sentence2 = self.dataframe.iloc[idx]['sentence2']
        sentence3 = self.dataframe.iloc[idx]['sentence3']
        label = self.dataframe.iloc[idx]['label']

        # Concatenate the three sentences into a single string
        combined_sentences = f"{sentence1} [SEP] {sentence2} [SEP] {sentence3}"

        # Tokenize the concatenated string
        encoding = self.tokenizer(
            combined_sentences,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )

        # Return the encoding and label
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# Create DataLoader for batch processing
def create_dataloaders(train_df, test_df, batch_size=8):
    train_dataset = ReasonPredictionDataset(train_df)
    test_dataset = ReasonPredictionDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# Define training function
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        # Move the batch to the device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Average Training Loss: {avg_loss}')

# Define evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Main function
def main():
    # Set seed
    seed_value = 42
    set_seed(seed_value)

    # Load datasets
    df_train = pd.read_csv('data/task2_train.csv')
    df_val = pd.read_csv('data/task2_validation.csv')
    df_test = pd.read_csv('data/task2_test.csv')

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("asafaya/bert-large-arabic", num_labels=3)

    # Determine device
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(df_train, df_test, batch_size=8)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training and evaluation
    epochs = 1
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train(model, train_loader, optimizer, device)
        evaluate(model, test_loader, device)

    # Final evaluation
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
