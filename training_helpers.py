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
        start_labels = inputs["start_positions"].to(outputs.start_logits.device)
        end_labels = inputs["end_positions"].to(outputs.end_logits.device)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        start_conf = start_probs.gather(1, start_labels.unsqueeze(-1)).squeeze(-1)
        end_conf = end_probs.gather(1, end_labels.unsqueeze(-1)).squeeze(-1)

        for i, ex_id in enumerate(example_ids):
            total_loss = -torch.log(start_conf[i] + 1e-12) - torch.log(end_conf[i] + 1e-12)
            confidence = ((start_conf[i] + end_conf[i]) / 2).item()
            self.example_losses[ex_id].append(total_loss.item())
            self.example_confidences[ex_id].append(confidence)
        
        

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