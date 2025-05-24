import numpy as np
import json
import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from torch.utils.data import DataLoader, Dataset
from scipy import signal
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchinfo import summary

"""
Data Augmentation Model
THis model make a forgery data from original genuine Handwriting data
"""

# Dataset Load
class SignatureDataset(Dataset):
    def __init__(self, json_path, label):
        """Initialize the dataset with the path to the JSON file.
        Args:
            json_path (str): Path to the JSON file containing the dataset.
        """
        super().__init__()
        self.device = torch.device("mps")
        
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)
        
        self.samples = self._process_data()
        self.labels = torch.tensor(label)
        #self.labels = torch.tensor(self.raw_data['writerCode'])

    def _process_data(self):
        processed_data = []
        for sample in self.raw_data['samples']:
            strokes = sample['strokes']

            sequence = []
            global_time = 0.0

            # Get Last Stroke TimeOffset
            stroke_durations = [p['points'][-1]['timeOffset'] for p in strokes]

            for i, points in enumerate(strokes):
                for point in points['points']:
                    features = [
                        point['x'],
                        point['y'],
                        point['force'],
                        point['altitude'],
                        point['azimuth'],
                        global_time + point['timeOffset'],
                    ]
                    sequence.append(features)
                global_time += stroke_durations[i]

            if sequence:
                # Fix length to 96
                sequence = signal.resample(sequence, 96)

                # Normalize
                sequence = self._normalize(np.array(sequence))

                tensor = torch.tensor(sequence, dtype=torch.float32)
                tensor = tensor.to(self.device)

                processed_data.append(tensor)
        
        return processed_data

    def _normalize(self, sequence):
        norm_seq = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            min_val = np.min(sequence[:, i])
            max_val = np.max(sequence[:, i])
            if max_val > min_val:
                norm_seq[:, i] = (sequence[:, i] - min_val) / (max_val - min_val)
            else: norm_seq[:, i] = 0.5

        return norm_seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample, self.labels


# Liquid Neural Network.

class HandwritingLNNAttention(nn.Module):
    def __init__(self, num_classes, seq_length=96, input_size=6):
        super().__init__()
        self.normalize = nn.LayerNorm(input_size)

        # Liquid Neural Network with NCP wiring
        self.wiring = AutoNCP(128, 64)
        self.ltc = LTC(input_size, self.wiring, batch_first=True)

        # Temporal Attention
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.normalize(x)

        outputs, _ = self.ltc(x)

        attn_weights = torch.softmax(self.attention(outputs), dim=1)
        context = torch.sum(attn_weights * outputs, dim=1)

        return self.classifier(context)

class LitHandwritingModel(pl.LightningModule):
    def __init__(self, num_classes, input_size, learning_rate=1e-3):
        super().__init__()
        
        self.model = HandwritingLNNAttention(num_classes, input_size=6)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Data Load Test
if __name__ == "__main__":
    # Model Test

    # Hyperparameters
    batch_size = 64
    num_classes = 2
    seq_length = 96
    input_size = 6

    Dataset_class_a = SignatureDataset('/Users/haro/works/handwriting_authenticate/data/class_A.json', 0)
    Dataset_class_b = SignatureDataset('/Users/haro/works/handwriting_authenticate/data/class_A_b.json', 1)

    dataset = Dataset_class_a + Dataset_class_b

    print(len(dataset))

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [180, 20])

      # Load Dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
    # Initialize model and trainer
    model = LitHandwritingModel(num_classes=num_classes, input_size=input_size)
    trainer = pl.Trainer(max_epochs=500, accelerator='mps', devices=1, logger=True)
    
    summary(model, input_size=(batch_size, seq_length, input_size))

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    