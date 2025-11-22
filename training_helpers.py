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
        # store per-feature information
        self.example_losses = defaultdict(list)
        self.example_confidences = defaultdict(list)
        self.example_predictions = defaultdict(list)
        self.example_true_labels = {}
        self.feature_to_example = {}

    def log_batch(self, inputs, outputs):
        """Collects data per feature for later aggregation"""
        feature_ids = inputs["_feature_id"]
        unique_ids  = inputs["_unique_id"]

        start_labels = inputs["start_positions"].to(outputs.start_logits.device)
        end_labels   = inputs["end_positions"].to(outputs.end_logits.device)

        start_logits = outputs.start_logits
        end_logits   = outputs.end_logits

        # compute probabilities
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs   = torch.softmax(end_logits, dim=-1)

        joint_conf = (start_probs.gather(1, start_labels.unsqueeze(-1)).squeeze(-1) *
                    end_probs.gather(1, end_labels.unsqueeze(-1)).squeeze(-1)).clamp(min=1e-12)
        total_losses = -torch.log(joint_conf)

        # predicted positions
        pred_start = torch.argmax(start_logits, dim=-1)
        pred_end   = torch.argmax(end_logits, dim=-1)

        for i in range(inputs["input_ids"].size(0)):
            fid = int(feature_ids[i])
            uid = unique_ids[i]

            self.example_losses[fid].append(total_losses[i].item())
            self.example_confidences[fid].append(joint_conf[i].item())
            self.example_predictions[fid].append((pred_start[i].item(), pred_end[i].item()))
            self.example_true_labels[fid] = (start_labels[i].item(), end_labels[i].item())
            self.feature_to_example[fid] = uid

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Reset per-epoch storage
        self.example_losses.clear()
        self.example_confidences.clear()
        self.example_predictions.clear()
        self.example_true_labels.clear()
        self.feature_to_example.clear()

    def on_epoch_end(self, args, state, control, **kwargs):

        data_map = []

        for fid, uid in self.feature_to_example.items():
            # convert lists to tensors
            losses_tensor = torch.tensor(self.example_losses[fid])
            confidences_tensor = torch.tensor(self.example_confidences[fid])
            preds = self.example_predictions[fid]
            true_start, true_end = self.example_true_labels[fid]

            # correctness: fraction of predictions exactly matching true start/end
            correctness = float(sum(1 for p in preds if p == (true_start, true_end)) / len(preds))

            # variability and average confidence
            variability = float(torch.std(losses_tensor).item())
            avg_confidence = float(torch.mean(confidences_tensor).item())

            data_map.append({
                "feature_id": int(fid),
                "example_id": int(uid),
                "correctness": correctness,
                "variability": variability,
                "confidence": avg_confidence
            })

        # write per-epoch JSONL file
        out_file = f"data_map_epoch_{int(state.epoch)}.jsonl"
        with open(out_file, "w") as f:
            for entry in data_map:
                f.write(json.dumps(entry) + "\n")

        print(f"[DataMapCallback] Wrote {len(data_map)} features to {out_file}")


def collate_fn_with_ids(batch):

    collated = default_data_collator(batch)

    # preserve metadata for callback
    collated["_unique_id"] = [ex["unique_id"] for ex in batch]
    
    # Keep feature_id as a tensor since it's an integer index, 
    # but check its type if it also caused errors previously.
    collated["_feature_id"] = torch.tensor([ex["feature_id"] for ex in batch])

    return collated