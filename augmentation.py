import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchio as tio
from scipy import signal



"""
Data Augmentation Model
THis model make a forgery data from original genuine Handwriting data
"""

# Dataset Load
class SignatureDataset(Dataset):
    def __init__(self, json_path):
        """Initialize the dataset with the path to the JSON file.
        Args:
            json_path (str): Path to the JSON file containing the dataset.
        """
        super().__init__()
        self.device = torch.device("mps")
        
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)
        
        self.samples = self._process_data()

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
                        #point['altitude'],
                        #point['azimuth'],
                        #global_time + point['timeOffset'],
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
        return sample


# Data Load Test
if __name__ == "__main__":
    dataset = SignatureDataset('/Users/haro/works/handwriting_authenticate/data/signature_data.json')

    print(f"Dataset contains {len(dataset)} signatures")
    
    sample = dataset[0]

    print(f"Sample tensor shape: {sample.shape}")  # [num_points, 4]
    print(f"Features (x, y, force, time):\n{sample[:5]}")

    