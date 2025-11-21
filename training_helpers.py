from transformers import TrainerCallback
from collections import defaultdict
import torch

class DataMapCallback(TrainerCallback):

    def __init__(self):
        self.loss_history = defaultdict(list)   # example_id → list of losses across epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch

    def on_step_end(self, args, state, control, **kwargs):
        """
        kwargs contains:
            - model
            - inputs (batch)
            - outputs (model outputs for the batch with reduced loss)
        """
        model = kwargs["model"]
        inputs = kwargs["inputs"]

        # Recompute forward pass but with reduction='none' so we get per-example loss
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=False, return_dict=True)
        
        # logits → compute individual loss manually
        # QA uses start/end logits → we compute cross-entropy loss per example
        start_loss = torch.nn.functional.cross_entropy(
            out.start_logits, inputs["start_positions"], reduction="none"
        )
        end_loss = torch.nn.functional.cross_entropy(
            out.end_logits, inputs["end_positions"], reduction="none"
        )

        batch_losses = (start_loss + end_loss) / 2

        # Record losses for every example
        for id_value, loss_val in zip(inputs["id"], batch_losses.detach().cpu().tolist()):
            self.loss_history[id_value].append(loss_val)
