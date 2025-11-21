import json
import os
import uuid
from collections import defaultdict

import torch

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class DataMapCallback(TrainerCallback):

    def __init__(self, output_dir="datamap_results"):
        super().__init__()
        self.example_losses = defaultdict(list)  # example_id -> list of losses
        self.example_confidences = defaultdict(list)  # example_id -> list of correct class probs

    def log_batch(self, inputs, outputs):
        # This will be called from the Trainer

        print("Logging batch...")

        example_ids = inputs.get("example_id", list(range(inputs["input_ids"].size(0))))
        start_labels = inputs["start_positions"]
        end_labels = inputs["end_positions"]

        

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        # print("On step end")

        # # for key, value in kwargs.items():
        # #     print(f"Key: {key}")

        # # print(f"Args: {args}")
        # # print(f"State: {state}")
        # # print(f"Control: {control}")

        # print(f"Model: {kwargs["model"]}")

        # # Trainer instance may not be passed directly; we can access via model.trainer
        # trainer = kwargs.get("model").trainer
        # if not trainer:
        #     return

        # inputs = getattr(trainer, "_current_inputs", None)
        # outputs = getattr(trainer, "_current_outputs", None)

        # print(f"Inputs: {inputs}")
        # print(f"Outputs: {outputs}")