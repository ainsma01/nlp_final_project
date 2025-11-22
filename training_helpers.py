import json
import os
import uuid
import torch.nn.functional as F
from collections import defaultdict
from transformers import default_data_collator
import torch

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class DataMapCallback(TrainerCallback):

    def __init__(self, output_dir="data_maps"):
        super().__init__()
        self.output_dir = output_dir
        self.example_losses = defaultdict(list)
        self.example_confidences = defaultdict(list)
        self.example_correct = defaultdict(list)

    def log_batch(self, inputs, outputs):
        """
        Log per-feature metrics directly keyed by unique example ID
        """
        # Move labels to the same device as logits
        start_labels = inputs['start_positions'].to(outputs.start_logits.device)
        end_labels = inputs['end_positions'].to(outputs.end_logits.device)
        unique_ids = [str(uid) for uid in inputs['unique_id']]

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute per-feature cross-entropy loss
        start_loss = F.cross_entropy(start_logits, start_labels, reduction='none')
        end_loss = F.cross_entropy(end_logits, end_labels, reduction='none')
        per_feature_loss = start_loss + end_loss

        # Compute confidence per feature (joint probability of true start and end)
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        joint_conf = (start_probs.gather(1, start_labels.unsqueeze(-1)).squeeze(-1) *
                      end_probs.gather(1, end_labels.unsqueeze(-1)).squeeze(-1)).clamp(min=1e-12)

        # Compute correctness (predicted == true)
        pred_start = torch.argmax(start_logits, dim=-1)
        pred_end = torch.argmax(end_logits, dim=-1)
        correct = ((pred_start == start_labels) & (pred_end == end_labels)).float()

        for i, uid in enumerate(unique_ids):
            self.example_losses[uid].append(per_feature_loss[i].item())
            self.example_confidences[uid].append(joint_conf[i].item())
            self.example_correct[uid].append(correct[i].item())

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Reset per-epoch storage
        self.example_losses.clear()
        self.example_confidences.clear()
        self.example_correct.clear()

    def on_epoch_end(self, args, state, control, **kwargs):
        data_map = []

        for uid in self.example_losses.keys():
            avg_loss = sum(self.example_losses[uid]) / len(self.example_losses[uid])
            avg_conf = sum(self.example_confidences[uid]) / len(self.example_confidences[uid])
            avg_correct = sum(self.example_correct[uid]) / len(self.example_correct[uid])

            data_map.append({
                "example_id": uid,
                "loss": avg_loss,
                "confidence": avg_conf,
                "correctness": avg_correct
            })

        out_file = f"{self.output_dir}/data_map_epoch_{int(state.epoch)}.jsonl"
        with open(out_file, "w") as f:
            for entry in data_map:
                f.write(json.dumps(entry) + "\n")

        print(f"[SimpleDataMapCallback] Wrote {len(data_map)} examples to {out_file}")


def collate_fn_with_ids(batch):
    collated = default_data_collator(batch)
    collated['unique_id'] = [ex['unique_id'] for ex in batch]  # list of strings
    return collated