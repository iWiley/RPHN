import os
import random
import numpy as np
import torch

def seed_everything(seed=42, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    # Keep MPS runs reproducible when available.
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(deterministic)
        if deterministic:
             os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except AttributeError:
        pass
    
def c_index_metric(risk: torch.Tensor, events: torch.Tensor, times: torch.Tensor) -> float:
    """Vectorized C-index on torch tensors without forcing CPU conversion."""
    risk = risk.view(-1)
    events = events.type(torch.bool).view(-1)
    times = times.view(-1)

    time_diff = times.unsqueeze(1) - times.unsqueeze(0)
    time_mask = time_diff < 0
    event_mask = events.unsqueeze(1)
    comparable_mask = time_mask & event_mask

    if not comparable_mask.any():
        return 0.5

    risk_diff = risk.unsqueeze(1) - risk.unsqueeze(0)
    concordant = (risk_diff > 0).float()
    ties = (risk_diff == 0).float()
    concordant_score = (concordant * comparable_mask.float()).sum()
    ties_score = 0.5 * (ties * comparable_mask.float()).sum()
    total_comparable = comparable_mask.float().sum()

    return ((concordant_score + ties_score) / total_comparable).item()
