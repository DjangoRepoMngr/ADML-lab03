"""Small, optional W&B helper that mirrors the slides. Safe to import if wandb
is not installed or you don’t want to use it.
"""
from typing import Optional

class WB:
    def __init__(self, use_wandb: bool = False, project: Optional[str] = None, config: Optional[dict] = None):
        self.enabled = False
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(project=project or "adml-lab3", config=config or {})
                self.enabled = True
            except Exception:
                # If wandb is missing or fails to init, just run silently.
                self.enabled = False
                self.wandb = None

    def log(self, metrics: dict):
        if self.enabled and self.wandb is not None:
            self.wandb.log(metrics)

    def finish(self):
        if self.enabled and self.wandb is not None:
            self.wandb.finish()