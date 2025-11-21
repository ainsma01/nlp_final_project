import json
import os
import uuid
from collections import defaultdict

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class DataMapCallback(TrainerCallback):
    def __init__(self, args: TrainingArguments):
        super().__init__()
        self.args = args

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        print("On step end")

        for key, value in kwargs.items():
            print(f"{key}: {value}")