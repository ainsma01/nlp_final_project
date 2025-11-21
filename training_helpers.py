import json
import os
import uuid
from collections import defaultdict
from transformers import default_data_collator
import torch

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class DataMapCallback(TrainerCallback):

    def __init__(self, output_dir="datamap_results"):
        super().__init__()
        self.example_losses = defaultdict(list)  # example_id -> list of losses
        self.example_confidences = defaultdict(list)  # example_id -> list of correct class probs

    def log_batch(self, inputs, outputs):
        # feature-level example IDs
        feature_ids = inputs["_feature_id"]
        unique_ids = inputs["_unique_id"]

        print(f"feature_ids{feature_ids}")

        print(f"unique_ids: {unique_ids}")

        example_ids = inputs.get("example_id", list(range(inputs["input_ids"].size(0))))
        
        # move labels to correct device
        start_labels = inputs["start_positions"].to(outputs.start_logits.device)
        end_labels = inputs["end_positions"].to(outputs.end_logits.device)

        # logits
        start_logits = outputs.start_logits       # shape: (batch, seq_len)
        end_logits   = outputs.end_logits         # shape: (batch, seq_len)

        # convert logits → probabilities
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs   = torch.softmax(end_logits,   dim=-1)

        # probability assigned to the TRUE start and TRUE end
        start_conf = start_probs.gather(1, start_labels.unsqueeze(-1)).squeeze(-1)
        end_conf   = end_probs.gather(1,   end_labels.unsqueeze(-1)).squeeze(-1)

        # NEW: joint probability = P(start)*P(end)
        joint_conf = (start_conf * end_conf).clamp(min=1e-12)

        # compute loss = -log P(start) - log P(end)
        # equivalent to -log(joint_conf)
        total_losses = -torch.log(joint_conf)

        # store results per example
        for i, ex_id in enumerate(example_ids):
            ex_id = int(ex_id)

            loss_val = total_losses[i].item()
            conf_val = joint_conf[i].item()

            self.example_losses[ex_id].append(loss_val)
            self.example_confidences[ex_id].append(conf_val)

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Reset storage for this epoch only
        self.example_losses = defaultdict(list)
        self.example_confidences = defaultdict(list)

    def on_epoch_end(self, args, state, control, **kwargs):

        epoch = int(state.epoch)
        output_path = os.path.join("datamap_results", f"dynamics_epoch_{epoch}.jsonl")

        os.makedirs("datamap_results", exist_ok=True)
        print(f"Saving Data Map statistics for epoch {epoch} → {output_path}")

        with open(output_path, "w") as f:
            for ex_id in sorted(self.example_losses.keys()):

                losses = self.example_losses[ex_id]             # list of per-step -log P(span)
                confs  = self.example_confidences[ex_id]        # list of per-step joint confidences

                # Aggregate correctly
                avg_loss = float(sum(losses) / len(losses))
                avg_conf = float(sum(confs) / len(confs))

                # Confidence variability across the epoch
                variability = float(torch.tensor(confs).std().item()) if len(confs) > 1 else 0.0

                # Write JSONL entry
                entry = {
                    "guid": int(ex_id),
                    "avg_loss": avg_loss,
                    "avg_confidence": avg_conf,
                    "confidence_variability": variability,
                    "all_losses": losses,
                    "all_confidences": confs,
                }

                f.write(json.dumps(entry) + "\n")

        return control

def collate_fn_with_ids(batch):
    collated = default_data_collator(batch)

    # preserve metadata for callback
    collated["_unique_id"] = torch.tensor([ex["unique_id"] for ex in batch])
    collated["_feature_id"] = torch.tensor([ex["feature_id"] for ex in batch])

    print(f"Hello mr data collector: {collated["_unique_id"]}")

    return collated