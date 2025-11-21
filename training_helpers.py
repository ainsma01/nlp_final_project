import json
import os
import uuid
from collections import defaultdict

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class DataMapCallback(TrainerCallback):
    """
    A custom callback to track training dynamics (loss and epoch) 
    for individual data samples to compute a Data Map.
    
    It assumes the preprocessed features contain a 'unique_id' field.
    """
    def __init__(self, output_dir="datamap_results"):
        """
        Initializes the storage for training dynamics.
        
        Args:
            output_dir (str): Directory where the final JSON results will be saved.
        """
        super().__init__()
        self.output_dir = output_dir
        # Store results as: {unique_id: {losses: [float], epochs: [float], total_steps: int}}
        self.datamap_results = defaultdict(lambda: {'losses': [], 'epochs': [], 'total_steps': 0})
        self.log_history = []
        print(f"DataMapCallback initialized. Results will be saved to '{output_dir}/datamap_results.json'")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        print("On step end")

        for key, value in kwargs.items():
            print(f"{key}: {value}")