from transformers import TrainerCallback
from collections import defaultdict
import torch, numpy as np

class TrainingDynamicsCallback(TrainerCallback):

    def __init__(self):
        self.dynamics = defaultdict(list)
        self.epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Logging from callback Epoch {self.epoch}")
        self.epoch = int(state.epoch)

    def on_prediction_step(self, args, state, control, outputs, **inputs):
        # outputs: model predictions
        # inputs: batch inputs including id
        print("Executing on prediction step")
        start_logits = outputs.start_logits.detach().cpu()
        end_logits   = outputs.end_logits.detach().cpu()

        batch_ids = inputs["id"].detach().cpu()
        gold_start = inputs["start_positions"].detach().cpu()
        gold_end   = inputs["end_positions"].detach().cpu()

        for i, ex_id in enumerate(batch_ids):
            s_logit = start_logits[i, gold_start[i]].item()
            e_logit = end_logits[i, gold_end[i]].item()

            pred_s = torch.argmax(start_logits[i]).item()
            pred_e = torch.argmax(end_logits[i]).item()
            correct = (pred_s == gold_start[i]) and (pred_e == gold_end[i])

            self.dynamics[int(ex_id)].append({
                "epoch": self.epoch,
                "start_logit": s_logit,
                "end_logit": e_logit,
                "correct": bool(correct)
            })
