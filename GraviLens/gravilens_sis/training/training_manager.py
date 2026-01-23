# %% ============================================================================
# SECTION 3: TRIAL MANAGEMENT
# ============================================================================

class SimpleTrialManager:
    """Lightweight checkpoint manager for experiment tracking"""
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

    def create_trial(self, trial_name: str, config: Dict[str, Any], description: str = "") -> str:
        """Create a new trial folder"""
        trial_dir = self.base_dir / trial_name

        if trial_dir.exists():
            print(f"⚠️  Trial '{trial_name}' exists. Checkpoints will be overwritten.")
        else:
            trial_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created trial: {trial_name}")

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

        metadata_file = trial_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.registry['trials'][trial_name] = {
            'created': str(datetime.now()),
            'status': 'running',
            'folder': str(trial_dir)
        }
        self._save_registry()

        print(f"📁 Location: {trial_dir}")
        print(f"📝 Description: {description}\n")
        return trial_name

    def save_checkpoint(self, trial_name: str, model: nn.Module, optimizer,
                       epoch: int, metrics: Dict[str, float], save_optimizer: bool = True):
        """Save checkpoint"""
        trial_dir = self.base_dir / trial_name
        if not trial_dir.exists():
            raise ValueError(f"Trial '{trial_name}' does not exist.")

        metadata_file = trial_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': str(datetime.now())
        }

        if save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        checkpoint_file = trial_dir / f"{trial_name}.pth"
        torch.save(checkpoint, checkpoint_file)

        metadata['current_epoch'] = epoch
        metadata['last_updated'] = str(datetime.now())
        metadata['final_metrics'] = metrics

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.registry['trials'][trial_name]['status'] = 'running'
        self._save_registry()

        print(f"💾 Saved: {checkpoint_file.name} (Epoch {epoch})")

    def mark_complete(self, trial_name: str):
        """Mark trial as completed"""
        trial_dir = self.base_dir / trial_name
        metadata_file = trial_dir / 'metadata.json'

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata['status'] = 'completed'
        metadata['completed_at'] = str(datetime.now())

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.registry['trials'][trial_name]['status'] = 'completed'
        self._save_registry()
        print(f"✅ Trial '{trial_name}' completed")

    def get_trial_info(self, trial_name: str) -> Dict[str, Any]:
        """Get trial metadata - FIXED: Added missing method"""
        trial_dir = self.base_dir / trial_name
        metadata_file = trial_dir / 'metadata.json'

        if not metadata_file.exists():
            raise ValueError(f"Trial '{trial_name}' does not exist or has no metadata.")

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def list_trials(self):
        """Print summary of all trials"""
        print(f"\n{'='*90}")
        print(f"{'TRIALS SUMMARY':^90}")
        print(f"{'='*90}")
        print(f"{'Trial Name':<35} {'Status':<12} {'Final Val Loss':<15} {'Final Val F1':<10}")
        print(f"{'-'*90}")

        for trial_name in self.registry['trials'].keys():
            trial_dir = self.base_dir / trial_name
            metadata_file = trial_dir / 'metadata.json'

            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                status = metadata.get('status', 'unknown')
                final_metrics = metadata.get('final_metrics', {})
                val_loss = final_metrics.get('val_loss', float('inf'))
                val_f1 = final_metrics.get('val_f1', 0.0)

                loss_str = f"{val_loss:.4f}" if val_loss != float('inf') else "N/A"
                f1_str = f"{val_f1:.4f}" if val_f1 > 0 else "N/A"

                print(f"{trial_name:<35} {status:<12} {loss_str:<15} {f1_str:<10}")

        print(f"{'='*90}\n")

