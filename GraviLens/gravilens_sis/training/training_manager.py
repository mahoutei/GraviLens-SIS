# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

class SimpleTrialManager:
    """Lightweight checkpoint and experiment-tracking manager."""
    def __init__(self, base_checkpoint_dir: str = './model_checkpoints'):
        self.base_dir = Path(base_checkpoint_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / 'trials_registry.json'
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'trials': {}, 'created': str(datetime.now())}

    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def create_trial(self, trial_name: str, config: Dict[str, Any],
                     description: str = "") -> str:
        trial_dir = self.base_dir / trial_name
        if trial_dir.exists():
            print(f"  Trial '{trial_name}' exists — checkpoints will be overwritten.")
        else:
            trial_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Created trial: {trial_name}")

        metadata = {
            'trial_name': trial_name,
            'created': str(datetime.now()),
            'description': description,
            'config': config,
            'status': 'running',
            'current_epoch': 0,
            'final_metrics': {},
            'last_updated': str(datetime.now())
        }
        with open(trial_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.registry['trials'][trial_name] = {
            'created': str(datetime.now()),
            'status': 'running',
            'folder': str(trial_dir)
        }
        self._save_registry()
        print(f"  Location: {trial_dir}\n  Description: {description}\n")
        return trial_name

    def save_checkpoint(self, trial_name: str, model: nn.Module, optimizer,
                        epoch: int, metrics: Dict[str, float],
                        save_optimizer: bool = True):
        trial_dir = self.base_dir / trial_name
        if not trial_dir.exists():
            raise ValueError(f"Trial '{trial_name}' does not exist.")

        with open(trial_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': str(datetime.now())
        }
        if save_optimizer:
            try:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            except AttributeError:
                pass

        torch.save(checkpoint, trial_dir / f"{trial_name}.pth")

        metadata['current_epoch'] = epoch
        metadata['last_updated'] = str(datetime.now())
        metadata['final_metrics'] = metrics
        with open(trial_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.registry['trials'][trial_name]['status'] = 'running'
        self._save_registry()
        print(f"  Saved checkpoint (Epoch {epoch})")

    def mark_complete(self, trial_name: str):
        trial_dir = self.base_dir / trial_name
        with open(trial_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        metadata['status'] = 'completed'
        metadata['completed_at'] = str(datetime.now())
        with open(trial_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        self.registry['trials'][trial_name]['status'] = 'completed'
        self._save_registry()
        print(f"  Trial '{trial_name}' completed.")

    def get_trial_info(self, trial_name: str) -> Dict[str, Any]:
        metadata_file = self.base_dir / trial_name / 'metadata.json'
        if not metadata_file.exists():
            raise ValueError(f"Trial '{trial_name}' has no metadata.")
        with open(metadata_file, 'r') as f:
            return json.load(f)

    def list_trials(self):
        print(f"\n{'='*90}")
        print(f"{'TRIALS SUMMARY':^90}")
        print(f"{'='*90}")
        print(f"{'Trial Name':<40} {'Status':<12} {'Val Loss':<12} {'Val F1':<10}")
        print(f"{'-'*90}")
        for name in self.registry['trials']:
            mf = self.base_dir / name / 'metadata.json'
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                fm = m.get('final_metrics', {})
                vl = fm.get('val_loss', float('inf'))
                vf = fm.get('val_f1', 0.0)
                print(f"{name:<40} {m.get('status','?'):<12} "
                      f"{vl if vl==float('inf') else f'{vl:.4f}':<12} "
                      f"{vf if vf==0 else f'{vf:.4f}':<10}")
        print(f"{'='*90}\n")
