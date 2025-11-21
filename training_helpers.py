from transformers import TrainerCallback, Trainer
from collections import defaultdict
import torch, numpy as np

import torch
from transformers import TrainerCallback
from collections import defaultdict

class DataMapsCallback(TrainerCallback):
    
    def __init__(self):
        self.dynamics = defaultdict(list)
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch)

    def on_step_end(self, args, state, control, outputs=None, **kwargs):
        """
        Fires every training step.
        Requires a custom Trainer that returns (loss, outputs).
        """
        print("Executing on step end callback")
        if outputs is None:
            return
        
        # --- Extract logits ---
        start_logits = outputs.start_logits.detach().cpu()
        end_logits   = outputs.end_logits.detach().cpu()

        print(f"Start logits: {start_logits}")
        print(f"End logits: {end_logits}")

        # Access model inputs from kwargs
        inputs = kwargs.get("inputs", {})

        print(f"Inputs: {inputs}")
        
        example_ids  = inputs["id"].detach().cpu()
        gold_start   = inputs["start_positions"].detach().cpu()
        gold_end     = inputs["end_positions"].detach().cpu()

        print(f"Example IDs: {example_ids}")
        print(f"Gold start: {gold_start}")
        print(f"Gold end: {gold_end}")

        # --- Per-example logging ---
        batch_size = len(example_ids)
        for i in range(batch_size):
            ex_id = int(example_ids[i])

            gold_s = int(gold_start[i])
            gold_e = int(gold_end[i])

            pred_s = torch.argmax(start_logits[i]).item()
            pred_e = torch.argmax(end_logits[i]).item()

            correct = (pred_s == gold_s) and (pred_e == gold_e)

            print(f"Correct: {correct}")

            # log gold logits
            record = {
                "epoch": self.current_epoch,
                "start_logit": float(start_logits[i, gold_s]),
                "end_logit": float(end_logits[i, gold_e]),
                "correct": bool(correct),
            }

            self.dynamics[ex_id].append(record)


class DataMapsTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        loss = outputs.loss
        # return outputs so on_step_end can access logits
        return (loss, outputs) if return_outputs else loss