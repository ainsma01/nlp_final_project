import json
import os
import uuid
from collections import defaultdict

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class DataMapCallback(TrainerCallback):
    """
    A custom callback to track training dynamics (loss and epoch) 
    for individual data samples to compute a Data Map.
    
    It assumes the preprocessed features contain a 'unique_id' field.
    """
    def __init__(self, output_dir="datamap_results"):
        """
        Initializes the storage for training dynamics.
        
        Args:
            output_dir (str): Directory where the final JSON results will be saved.
        """
        super().__init__()
        self.output_dir = output_dir
        # Store results as: {unique_id: {losses: [float], epochs: [float], total_steps: int}}
        self.datamap_results = defaultdict(lambda: {'losses': [], 'epochs': [], 'total_steps': 0})
        self.log_history = []
        print(f"DataMapCallback initialized. Results will be saved to '{output_dir}/datamap_results.json'")

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Ensure the output directory exists before training starts.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_history = [] # Reset for safety

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Capture the logged loss and step information.
        The Trainer typically logs the average batch loss after N steps.
        We capture this log history here.
        """
        if logs is not None and 'loss' in logs:
            self.log_history.append({
                'loss': logs['loss'],
                'step': state.global_step,
                'epoch': state.epoch,
            })
            
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, tokenizer=None, optimizer=None, lr_scheduler=None, **kwargs):
        """
        After each training step, we need to associate the loss (from the last log) 
        with the samples in the current batch.
        
        IMPORTANT: This assumes that the last logged average loss (from on_log) 
        is a reasonable proxy for the loss of the samples in this training step.
        """
        # This is a critical point: The 'inputs' dictionary contains the batch data.
        # We need the unique IDs from this batch.
        inputs = kwargs.get('inputs')
        if not inputs or 'unique_id' not in inputs:
            # This is expected behavior if the input dict doesn't contain the IDs
            # (e.g., if the user is running evaluation or prediction steps).
            
            # --- FIX: Replaced state.is_local_main_process with args.local_rank check ---
            # local_rank is 0 for the main process in distributed training, and -1 otherwise.
            is_main_process = args.local_rank in [-1, 0] 
            if is_main_process and state.global_step % 100 == 0:
                # Log only periodically to avoid too much console noise
                print("Warning: 'unique_id' not found in inputs. Skipping datamap tracking for this step.")
            return

        current_loss = None
        current_epoch = None
        
        # 1. Find the most recent logged loss value
        # We search the log history for the loss corresponding to the current step or the most recent one.
        if self.log_history:
             # Find the latest loss logged *before or at* the current step
            last_log = next(
                (log for log in reversed(self.log_history) if log['step'] <= state.global_step),
                None
            )
            if last_log:
                current_loss = last_log['loss']
                current_epoch = last_log['epoch']
                
        if current_loss is None:
            # If no loss has been logged yet (e.g., very early in training)
            return

        # 2. Get the unique IDs from the batch
        unique_ids = inputs['unique_id']
        
        # 3. Store the metrics for each sample in the current batch
        for uid in unique_ids:
            uid_str = str(uid)
            self.datamap_results[uid_str]['losses'].append(current_loss)
            self.datamap_results[uid_str]['epochs'].append(current_epoch)
            self.datamap_results[uid_str]['total_steps'] += 1

    def on_train_end(self, args, state, control, **kwargs):
        """
        Save the final accumulated results at the end of training.
        """
        output_path = os.path.join(self.output_dir, "datamap_results.json")
        
        # Convert defaultdict to regular dict for saving
        final_results = dict(self.datamap_results)
        
        # Also compute mean and variance for convenience
        for uid, data in final_results.items():
            losses = data['losses']
            if losses:
                mean_loss = sum(losses) / len(losses)
                variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
                data['mean_loss'] = mean_loss
                data['loss_variance'] = variance

        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=4)
        
        print(f"\n--- Data Map Calculation Complete ---")
        print(f"Total samples tracked: {len(final_results)}")
        print(f"Results saved to: {output_path}")

def add_unique_id_to_features(examples):
    """
    Preprocessing function to add a unique 'unique_id' field to each 
    tokenized feature. This is the ID the callback will track.
    
    The user mentioned having a function to add an ID, so this is a placeholder 
    implementation for a batch of examples.
    """
    # Create a unique ID for each example/feature in the batch
    ids = [str(uuid.uuid4()) for _ in range(len(examples['input_ids']))]
    examples['unique_id'] = ids
    return examples