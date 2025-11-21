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

    def on_train_end(self, args, state, control, **kwargs):
        data_map = {}
        for ex_id in self.example_losses:
            losses = self.example_losses[ex_id]
            confidences = self.example_confidences[ex_id]
            data_map[ex_id] = {
                "losses": losses,
                "avg_loss": sum(losses)/len(losses),
                "confidence": sum(confidences)/len(confidences),
                "variability": torch.std(torch.tensor(confidences)).item()
            }

        with open("data_map.json", "w") as f:
            json.dump(data_map, f, indent=2)
        print("Data map saved to data_map.json")