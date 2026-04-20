import os
import argparse
import time
import re
import secrets
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.data.loader import create_dataloader
from src.eval_utils import MANUSCRIPT_ROOT, build_training_model, default_config_path, load_config
from src.utils.common import seed_everything, c_index_metric
from src.utils.losses import HybridSurvivalLoss

def _safe_name(name):
    return re.sub(r'[^0-9A-Za-z._-]+', '_', str(name)).strip('_') or 'split'


def _resolve_training_seed(raw_seed) -> tuple[int, str]:
    if raw_seed is None:
        return secrets.randbelow(2**31 - 1) + 1, 'random'
    seed = int(raw_seed)
    if seed == 0:
        return secrets.randbelow(2**31 - 1) + 1, 'random'
    return seed, 'fixed'

def _should_save_epoch_checkpoint(cfg: dict, epoch_num: int) -> bool:
    keep_epochs = cfg.get('save_epoch_checkpoints')
    if isinstance(keep_epochs, (list, tuple, set)):
        normalized = {int(x) for x in keep_epochs}
        if epoch_num in normalized:
            return True

    start = cfg.get('save_epoch_checkpoint_start')
    end = cfg.get('save_epoch_checkpoint_end')
    if start is not None and end is not None:
        return int(start) <= int(epoch_num) <= int(end)

    return False

class MetricsTracker:
    """Persist epoch summaries as a single flat CSV history."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.history = []
        self.save_path_csv = self.log_dir / "training_metrics.csv"

    def update(self, epoch_stats, is_best: bool = False):
        epoch_stats["is_best"] = bool(is_best)
        epoch_stats["timestamp"] = time.time()

        existing_idx = next((i for i, row in enumerate(self.history) if row["epoch"] == epoch_stats["epoch"]), None)
        if existing_idx is not None:
            self.history[existing_idx].update(epoch_stats)
        else:
            self.history.append(epoch_stats)

        pd.DataFrame(self.history).to_csv(self.save_path_csv, index=False)


def clear_saved_test_predictions(pred_dir: Path) -> None:
    if not pred_dir.exists():
        return
    for path in pred_dir.glob("epoch_*_test_*.csv"):
        path.unlink()


def save_epoch_predictions(preds, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _to_flat_numpy(values):
        if not values:
            return []
        return torch.cat(values).cpu().numpy().flatten() if isinstance(values[0], torch.Tensor) else values

    pd.DataFrame(
        {
            "PatientID": _to_flat_numpy(preds.get("names", [])),
            "OS_Pred": _to_flat_numpy(preds.get("os_s", [])),
            "OS_Event": _to_flat_numpy(preds.get("os_e", [])),
            "OS_Time": _to_flat_numpy(preds.get("os_t", [])),
            "RFS_Pred": _to_flat_numpy(preds.get("rfs_s", [])),
            "RFS_Event": _to_flat_numpy(preds.get("rfs_e", [])),
            "RFS_Time": _to_flat_numpy(preds.get("rfs_t", [])),
        }
    ).to_csv(save_path, index=False)


def build_runtime_context(cfg: dict, device: torch.device, seed: int) -> dict:
    batch_size = int(cfg.get("batch_size", 176))
    eval_batch_size = int(cfg.get("eval_batch_size", batch_size))
    loader_cfg = cfg.get("loader") or {}

    train_loader, train_len = create_dataloader(
        cfg.get("train"), batch_size, is_train=True, loader_cfg=loader_cfg
    )
    val_loader, val_len = create_dataloader(
        cfg.get("val"), eval_batch_size, is_train=False, loader_cfg=loader_cfg
    )

    test_loaders = []
    for test_cfg in cfg.get("test", []):
        test_loader, test_len = create_dataloader(
            test_cfg, eval_batch_size, is_train=False, loader_cfg=loader_cfg
        )
        test_loaders.append({"name": test_cfg["name"], "loader": test_loader, "length": test_len})

    model = build_training_model(cfg, device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    for param_group in optimizer.param_groups:
        param_group["target_lr"] = param_group["lr"]

    scheduler = None
    if cfg.get("use_plateau_scheduler", True) and val_loader:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = HybridSurvivalLoss(
        cca_weight=float(cfg.get("cca_weight", 0.1)),
        intra_decor_weight=float(cfg.get("intra_decor_weight", 0.05)),
        aux_weight=float(cfg.get("aux_weight", 1.0)),
        os_loss_weight=float(cfg.get("os_loss_weight", 1.0)),
        rfs_loss_weight=float(cfg.get("rfs_loss_weight", 1.0)),
    ).to(device)

    output_dir = cfg.get("output_dir", "results")
    output_root = Path(output_dir) if os.path.isabs(str(output_dir)) else (MANUSCRIPT_ROOT / output_dir)
    log_dir = output_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    return {
        "loader_cfg": loader_cfg,
        "train_loader": train_loader,
        "train_len": train_len,
        "val_loader": val_loader,
        "val_len": val_len,
        "test_loaders": test_loaders,
        "model": model,
        "optimizer": optimizer,
        "trainable_params": trainable_params,
        "scheduler": scheduler,
        "use_amp": use_amp,
        "scaler": scaler,
        "criterion": criterion,
        "log_dir": log_dir,
    }


def _move_batch_to_device(batch_obj, device, non_blocking=False):
    if torch.is_tensor(batch_obj):
        return batch_obj.to(device, non_blocking=non_blocking)
    if isinstance(batch_obj, list):
        return [x.to(device, non_blocking=non_blocking) for x in batch_obj]
    if isinstance(batch_obj, tuple):
        return tuple(x.to(device, non_blocking=non_blocking) for x in batch_obj)
    raise TypeError(f"Unsupported batch object type: {type(batch_obj).__name__}")


def _device_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class StepTimer:
    def __init__(self):
        self.totals = {
            'data_wait': 0.0,
            'h2d': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'optim': 0.0,
            'postprocess': 0.0,
            'metric': 0.0,
        }
        self.steps = 0

    def add(self, key, value):
        if key not in self.totals:
            self.totals[key] = 0.0
        self.totals[key] += float(value)

    def step(self):
        self.steps += 1

    def averages(self):
        denom = max(1, self.steps)
        return {k: v / denom for k, v in self.totals.items()}


class SectionTimer:
    def __init__(self):
        self.sections = {}

    def add(self, key, value):
        self.sections[key] = self.sections.get(key, 0.0) + float(value)

    def summary(self):
        return dict(self.sections)


def _format_timing_summary(prefix: str, summary: dict) -> str:
    ordered = ["data_wait", "h2d", "forward", "backward", "optim", "postprocess", "metric"]
    parts = [f"{k}={summary[k]:.3f}s" for k in ordered if k in summary]
    return f"{prefix}{' '.join(parts)}"


def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch, device, use_amp=False, wsi_anchor_momentum=0.99, timing_cfg=None):
    model.train()
    metrics = {'loss': 0, 'l_os': 0, 'l_rfs': 0, 'l_cca': 0, 'l_intra': 0, 'count': 0}
    preds = {k: [] for k in ['os_s', 'os_e', 'os_t', 'rfs_s', 'rfs_e', 'rfs_t']}
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False, dynamic_ncols=True)
    timing_cfg = timing_cfg or {}
    enable_timing = bool(timing_cfg.get('enable', False))
    print_every = max(1, int(timing_cfg.get('print_every', 20)))
    step_timer = StepTimer()

    loader_iter = iter(loader)
    step_idx = 0
    while True:
        fetch_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        step_timer.add('data_wait', time.perf_counter() - fetch_start)
        step_idx += 1
        if batch is None: continue
        wsi, pos, ct_vol, ct_masks, evt_os, tm_os, evt_rfs, tm_rfs, names = batch
        non_blocking = device.type == 'cuda'

        h2d_start = time.perf_counter()
        wsi = _move_batch_to_device(wsi, device, non_blocking=non_blocking)
        pos = _move_batch_to_device(pos, device, non_blocking=non_blocking)
        ct_vol = _move_batch_to_device(ct_vol, device, non_blocking=non_blocking)
        ct_masks = _move_batch_to_device(ct_masks, device, non_blocking=non_blocking)
        evt_os, tm_os, evt_rfs, tm_rfs = [x.to(device, non_blocking=non_blocking) for x in [evt_os, tm_os, evt_rfs, tm_rfs]]
        if enable_timing:
            _device_sync(device)
        step_timer.add('h2d', time.perf_counter() - h2d_start)
        
        optimizer.zero_grad()
        fw_start = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            os_score, rfs_score, out_dict = model(wsi, pos, ct_vol, ct_masks)
            l_total, l_dict = criterion(os_score, rfs_score, evt_os, tm_os, evt_rfs, tm_rfs, out_dict)
        if enable_timing:
            _device_sync(device)
        step_timer.add('forward', time.perf_counter() - fw_start)
        
        bw_start = time.perf_counter()
        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        if enable_timing:
            _device_sync(device)
        step_timer.add('backward', time.perf_counter() - bw_start)

        opt_start = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        if hasattr(model, 'apply_wsi_anchor_momentum'):
            model.apply_wsi_anchor_momentum(momentum=wsi_anchor_momentum)
        if enable_timing:
            _device_sync(device)
        step_timer.add('optim', time.perf_counter() - opt_start)
        step_timer.step()
        
        bs = len(names)
        metrics['loss'] += l_total.item() * bs
        metrics['l_os'] += l_dict.get('l_os', 0) * bs
        metrics['l_rfs'] += l_dict.get('l_rfs', 0) * bs
        metrics['l_cca'] += l_dict.get('l_cca', 0) * bs
        metrics['l_intra'] += l_dict.get('l_intra', 0) * bs
        metrics['count'] += bs

        avg_loss = metrics['loss'] / max(1, metrics['count'])
        progress.set_postfix(
            loss=f"{avg_loss:.3f}",
            os=f"{float(l_dict.get('l_os', 0.0)):.3f}",
            rfs=f"{float(l_dict.get('l_rfs', 0.0)):.3f}",
        )
        if enable_timing and step_idx % print_every == 0:
            avg_t = step_timer.averages()
            print(
                f"[Timing][E{epoch+1:03d} S{step_idx:04d}] "
                f"data={avg_t['data_wait']:.3f}s h2d={avg_t['h2d']:.3f}s "
                f"fw={avg_t['forward']:.3f}s bw={avg_t['backward']:.3f}s opt={avg_t['optim']:.3f}s"
            )
        
        for k, v in zip(['os_s', 'os_e', 'os_t', 'rfs_s', 'rfs_e', 'rfs_t'], [os_score.detach().float(), evt_os, tm_os, rfs_score.detach().float(), evt_rfs, tm_rfs]):
            preds[k].append(v)
    avg_losses = {k: v / max(1, metrics['count']) for k, v in metrics.items() if k != 'count'}
    tr_c_os = c_index_metric(torch.cat(preds['os_s']), torch.cat(preds['os_e']), torch.cat(preds['os_t']))
    tr_c_rfs = c_index_metric(torch.cat(preds['rfs_s']), torch.cat(preds['rfs_e']), torch.cat(preds['rfs_t']))
    timing_summary = step_timer.averages() if enable_timing else {}
    return avg_losses, tr_c_os, tr_c_rfs, timing_summary


def evaluate(model, loader, device, use_amp=False, criterion=None, timing_cfg=None):
    model.eval()
    preds = {k: [] for k in ['os_s', 'os_e', 'os_t', 'rfs_s', 'rfs_e', 'rfs_t', 'names']}
    total_loss, count = 0.0, 0
    timing_cfg = timing_cfg or {}
    enable_timing = bool(timing_cfg.get('enable', False))
    timer = StepTimer()

    loader_iter = iter(loader)
    with torch.no_grad():
        while True:
            fetch_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            timer.add('data_wait', time.perf_counter() - fetch_start)
            if batch is None: continue
            wsi, pos, ct_vol, ct_masks, evt_os, tm_os, evt_rfs, tm_rfs, names = batch
            non_blocking = device.type == 'cuda'

            h2d_start = time.perf_counter()
            wsi = _move_batch_to_device(wsi, device, non_blocking=non_blocking)
            pos = _move_batch_to_device(pos, device, non_blocking=non_blocking)
            ct_vol = _move_batch_to_device(ct_vol, device, non_blocking=non_blocking)
            ct_masks = _move_batch_to_device(ct_masks, device, non_blocking=non_blocking)
            evt_os_d, tm_os_d, evt_rfs_d, tm_rfs_d = [x.to(device, non_blocking=non_blocking) for x in [evt_os, tm_os, evt_rfs, tm_rfs]]
            if enable_timing:
                _device_sync(device)
            timer.add('h2d', time.perf_counter() - h2d_start)

            fw_start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                os_score, rfs_score, out_dict = model(wsi, pos, ct_vol, ct_masks)
                if criterion is not None:
                    l_total, _ = criterion(os_score, rfs_score, evt_os_d, tm_os_d, evt_rfs_d, tm_rfs_d, out_dict)
                    total_loss += l_total.item() * len(names)
                    count += len(names)
            if enable_timing:
                _device_sync(device)
            timer.add('forward', time.perf_counter() - fw_start)

            post_start = time.perf_counter()
            for k, v in zip(['os_s', 'os_e', 'os_t', 'rfs_s', 'rfs_e', 'rfs_t'], [os_score.detach().float(), evt_os_d, tm_os_d, rfs_score.detach().float(), evt_rfs_d, tm_rfs_d]):
                preds[k].append(v)
            preds['names'].extend(names)
            timer.add('postprocess', time.perf_counter() - post_start)
            timer.step()

    metric_start = time.perf_counter()
    c_os = c_index_metric(torch.cat(preds['os_s']), torch.cat(preds['os_e']), torch.cat(preds['os_t']))
    c_rfs = c_index_metric(torch.cat(preds['rfs_s']), torch.cat(preds['rfs_e']), torch.cat(preds['rfs_t']))
    timer.add('metric', time.perf_counter() - metric_start)
    timing_summary = timer.averages() if enable_timing else {}
    return c_os, c_rfs, total_loss / max(1, count) if criterion else None, preds, timing_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        help='Path to config file',
        default=str(default_config_path("default.yaml")),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed, seed_mode = _resolve_training_seed(cfg.get('seed'))
    deterministic = bool(cfg.get('deterministic', False))
    seed_everything(seed, deterministic=deterministic)
    print(f"[Reproducibility] seed={seed} ({seed_mode}) | deterministic={deterministic}")
    print(f"Using Device: {device}\n\n--- Cohort Initialization ---")

    runtime = build_runtime_context(cfg, device, seed)
    loader_cfg = runtime['loader_cfg']
    train_loader = runtime['train_loader']
    train_len = runtime['train_len']
    val_loader = runtime['val_loader']
    val_len = runtime['val_len']
    test_loaders = runtime['test_loaders']
    model = runtime['model']
    optimizer = runtime['optimizer']
    trainable_params = runtime['trainable_params']
    scheduler = runtime['scheduler']
    use_amp = runtime['use_amp']
    scaler = runtime['scaler']
    criterion = runtime['criterion']
    log_dir = runtime['log_dir']

    print(
        "[DataLoader] "
        f"num_workers={int(loader_cfg.get('num_workers', min(8, os.cpu_count() or 1)))} | "
        f"pin_memory={bool(loader_cfg.get('pin_memory', torch.cuda.is_available()))} | "
        f"cache_mode={str(loader_cfg.get('cache_mode', 'none'))} | "
        f"cache_workers={loader_cfg.get('cache_workers', 4)}"
    )

    print(f"Cohort Info: Train({train_len}), Val({val_len})")
    print("-----------------------------\n")

    total_cases = int(train_len + val_len + sum(item['length'] for item in test_loaders))

    print(f"[Resolved Inputs] total cases={total_cases}")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Optimizer] trainable parameters={num_trainable/1e6:.2f}M")
    print(f"  - Parameter Tensors: {len(trainable_params)}")

    warmup_epochs, warmup_start_factor = int(cfg.get('warmup_epochs', 0)), float(cfg.get('warmup_start_factor', 0.2))
    if warmup_epochs > 0:
        for pg in optimizer.param_groups:
            pg['lr'] = pg['target_lr'] * warmup_start_factor
    print(f"Results saved to: {log_dir}")
    
    tracker = MetricsTracker(log_dir)
    best_score, smoothed_val_score, patience_counter, best_epoch = -1.0, None, 0, 0
    save_epoch_preds, pred_dir = bool(cfg.get('save_epoch_predictions', False)), log_dir / 'predictions'
    ckpt_dir = log_dir / 'checkpoints'
    eval_test_each_epoch = bool(cfg.get('eval_test_each_epoch', False))
    eval_test_every = 1 if eval_test_each_epoch else int(cfg.get('eval_test_every', 0))
    eval_val_every = max(1, int(cfg.get('eval_val_every', 1)))
    eval_test_on_best = bool(cfg.get('eval_test_on_best', True))
    early_stop_patience = int(cfg.get('early_stop_patience', 0))
    stop_training = False
    timing_cfg = cfg.get('timing') or {}
    enable_epoch_timing = bool(timing_cfg.get('enable', False))

    def run_test_evaluations(row, epoch_num, epoch_sections, timing_prefix):
        summaries = []
        for t_item in test_loaders:
            t_os, t_rfs, _, t_preds, test_timing = evaluate(model, t_item['loader'], device, use_amp, timing_cfg=timing_cfg)
            s_name = _safe_name(t_item['name'])
            row.update({
                f"test_{s_name}_os": float(t_os),
                f"test_{s_name}_rfs": float(t_rfs),
                f"test_{s_name}_sum": float(t_os + t_rfs),
            })
            summaries.append(f"{t_item['name']}={t_os:.3f}/{t_rfs:.3f}")
            if test_timing:
                print("   " + _format_timing_summary(f"[{timing_prefix}][{t_item['name']}] ", test_timing))
            if save_epoch_preds:
                save_test_pred_start = time.perf_counter()
                save_epoch_predictions(t_preds, str(pred_dir / f"epoch_{epoch_num:03d}_test_{s_name}.csv"))
                epoch_sections.add('save_test_predictions', time.perf_counter() - save_test_pred_start)
        return summaries

    for epoch in range(cfg.get('epochs', 100)):
        epoch_sections = SectionTimer()
        if 0 < warmup_epochs > epoch:
            factor = warmup_start_factor + (1.0 - warmup_start_factor) * ((epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = pg['target_lr'] * factor

        train_epoch_start = time.perf_counter()
        train_loss_dict, c_os, c_rfs, timing_summary = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, epoch, device, 
            use_amp, float(cfg.get('wsi_anchor_momentum', 0.99)), timing_cfg=timing_cfg
        )
        epoch_sections.add('train_epoch', time.perf_counter() - train_epoch_start)
        
        loss = train_loss_dict['loss']

        row = {
            'epoch': epoch + 1, 'train_loss': float(loss), 
            'train_os': float(c_os), 'train_rfs': float(c_rfs), 'train_sum': float(c_os + c_rfs),
            **{f"train_{k}": float(v) for k, v in train_loss_dict.items()},
        }

        print(f"E{epoch+1:03d} | L:{loss:.3f} | OS_L:{train_loss_dict['l_os']:.3f}, RFS_L:{train_loss_dict['l_rfs']:.3f} | "
              f"CCA:{train_loss_dict['l_cca']:.3f} | "
              f"TR_C: OS={c_os:.3f}, RFS={c_rfs:.3f}")
        if timing_summary:
            print(
                f"   [Epoch Timing] data={timing_summary['data_wait']:.3f}s "
                f"h2d={timing_summary['h2d']:.3f}s fw={timing_summary['forward']:.3f}s "
                f"bw={timing_summary['backward']:.3f}s opt={timing_summary['optim']:.3f}s"
            )

        is_best_epoch = False
        ran_test_this_epoch = False
        run_val_this_epoch = bool(val_loader) and (((epoch + 1) % eval_val_every) == 0 or (epoch + 1) == int(cfg.get('epochs', 100)))
        if run_val_this_epoch:
            val_epoch_start = time.perf_counter()
            v_os, v_rfs, v_loss, val_preds, val_timing = evaluate(model, val_loader, device, use_amp, criterion, timing_cfg=timing_cfg)
            epoch_sections.add('val_epoch', time.perf_counter() - val_epoch_start)
            if val_timing:
                print("   " + _format_timing_summary("[Val Timing] ", val_timing))
            score = v_os + v_rfs
            row.update({'val_loss': float(v_loss), 'val_os': float(v_os), 'val_rfs': float(v_rfs), 'val_sum': float(score)})
            
            if scheduler and epoch >= warmup_epochs: scheduler.step(score)
            
            smoothed_val_score = score if smoothed_val_score is None else 0.4 * smoothed_val_score + 0.6 * score
            row['val_smoothed'] = float(smoothed_val_score)

            if smoothed_val_score > best_score + 1e-3:
                best_score, best_epoch, patience_counter, is_best_epoch = smoothed_val_score, epoch + 1, 0, True
                save_best_start = time.perf_counter()
                torch.save(model.state_dict(), log_dir / 'best_model.pth')
                epoch_sections.add('save_best_model', time.perf_counter() - save_best_start)
                print(f"   [Snapshot] \033[92m★ BEST MODEL SAVED (E{best_epoch}) ★\033[0m -> Smoothed: {best_score:.4f} (Raw: {score:.4f}, Val_Loss: {v_loss:.3f})")
                if test_loaders and eval_test_on_best:
                    best_test_start = time.perf_counter()
                    if save_epoch_preds and eval_test_every <= 0:
                        clear_pred_start = time.perf_counter()
                        clear_saved_test_predictions(pred_dir)
                        epoch_sections.add('clear_test_predictions', time.perf_counter() - clear_pred_start)
                    best_test_summary = run_test_evaluations(row, epoch + 1, epoch_sections, "Best Test Timing")
                    epoch_sections.add('best_test_eval', time.perf_counter() - best_test_start)
                    print(f"   [Best Test] {' | '.join(best_test_summary)}")
                    ran_test_this_epoch = True
            else:
                patience_counter += 1
                print(f"   [Patience] No strict improvement for {patience_counter} epochs (Best was E{best_epoch}: {best_score:.4f})")
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    stop_training = True

            if save_epoch_preds:
                save_val_pred_start = time.perf_counter()
                save_epoch_predictions(val_preds, str(pred_dir / f"epoch_{epoch + 1:03d}_val.csv"))
                epoch_sections.add('save_val_predictions', time.perf_counter() - save_val_pred_start)
        elif val_loader:
            row.update({'val_loss': np.nan, 'val_os': np.nan, 'val_rfs': np.nan, 'val_sum': np.nan, 'val_smoothed': float(smoothed_val_score) if smoothed_val_score is not None else np.nan})
            print(f"   [Validation] skipped at epoch {epoch + 1} (eval_val_every={eval_val_every})")

        epoch_num = epoch + 1
        if _should_save_epoch_checkpoint(cfg, epoch_num):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_payload = {
                'epoch': epoch_num,
                'seed': seed,
                'seed_mode': seed_mode,
                'config_path': str(args.config),
                'model_state_dict': model.state_dict(),
                'row': row,
            }
            save_ckpt_start = time.perf_counter()
            torch.save(ckpt_payload, ckpt_dir / f"epoch_{epoch_num:03d}.pth")
            epoch_sections.add('save_checkpoint', time.perf_counter() - save_ckpt_start)
            print(f"   [Snapshot] Saved scheduled checkpoint: E{epoch_num:03d}")

        test_summary = ""
        if test_loaders and eval_test_every > 0 and not ran_test_this_epoch and (epoch + 1) % eval_test_every == 0:
            periodic_test_start = time.perf_counter()
            periodic_test_summary = run_test_evaluations(row, epoch + 1, epoch_sections, "Periodic Test Timing")
            test_summary = " | TEST: " + " | ".join(periodic_test_summary)
            epoch_sections.add('periodic_test_eval', time.perf_counter() - periodic_test_start)
        
        tracker_start = time.perf_counter()
        tracker.update(row, is_best=is_best_epoch)
        epoch_sections.add('tracker_update', time.perf_counter() - tracker_start)
        val_loss_str = f"VL:{row['val_loss']:.3f} | " if 'val_loss' in row else ""
        summary_line = (
            f"E{epoch + 1:03d} | "
            f"L:{loss:.3f} | "
            f"TR: OS={c_os:.3f}, RFS={c_rfs:.3f} | "
            f"{val_loss_str}"
            f"VAL: OS={row.get('val_os', 0):.3f}, RFS={row.get('val_rfs', 0):.3f}"
            f"{test_summary}"
        )
        print(summary_line)
        if enable_epoch_timing:
            section_text = " ".join(f"{k}={v:.3f}s" for k, v in epoch_sections.summary().items())
            print(f"   [Epoch Sections] {section_text}")
        if stop_training:
            print(f"[Early Stop] Stopping at epoch {epoch + 1} after {patience_counter} non-improving validation epochs.")
            break
    
    if enable_epoch_timing:
        print("[Finalize Timing] tracker history has been flushed to CSV.")

if __name__ == "__main__": main()
