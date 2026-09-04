#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/rfd3_exec.sh" "$0" "$@"'

import json
import logging
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import rootutils
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from foundry.utils.logging import suppress_warnings
from foundry.utils.weights import (
    CheckpointConfig,
    ParameterFreezingConfig,
    WeightLoadingConfig,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv(override=True)

_config_path = os.path.join(os.environ["PROJECT_ROOT"], "models/rfd3/configs")
_spawning_process_logger = logging.getLogger(__name__)


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "model"):
        return model.model
    return model


def _import_lora_utils() -> tuple[Callable, Callable]:
    try:
        from rfd3.lora import inject_lora_into_model, count_trainable_parameters

        return inject_lora_into_model, count_trainable_parameters
    except ModuleNotFoundError:
        from lora import inject_lora_into_model, count_trainable_parameters

        return inject_lora_into_model, count_trainable_parameters


def _import_nmf_utils() -> tuple[Callable, Callable]:
    try:
        from rfd3.nmf import inject_nmf_into_model, count_trainable_parameters

        return inject_nmf_into_model, count_trainable_parameters
    except ModuleNotFoundError:
        from nmf import inject_nmf_into_model, count_trainable_parameters

        return inject_nmf_into_model, count_trainable_parameters


def _apply_lora_if_enabled(cfg: DictConfig, trainer) -> None:
    if not (cfg.get("lora", None) and cfg.lora.enabled):
        return

    inject_lora_into_model, count_trainable_parameters = _import_lora_utils()
    model = trainer.state["model"]
    base_model = _unwrap_model(model)

    ranked_logger = logging.getLogger(__name__)
    ranked_logger.info(f"Applying LoRA to base model type: {type(base_model).__name__}")

    lora_root_model = deepcopy(base_model)
    if not hasattr(lora_root_model, "diffusion_module"):
        raise AttributeError(
            f"Expected the base model to expose `diffusion_module`, but got {type(lora_root_model).__name__}."
        )

    lora_root_model.diffusion_module = inject_lora_into_model(
        lora_root_model.diffusion_module,
        target_keywords=cfg.lora.target_keywords,
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        freeze_all=True,
    )

    if cfg.lora.apply_to_token_initializer:
        if not hasattr(lora_root_model, "token_initializer"):
            raise AttributeError(
                f"Expected the base model to expose `token_initializer`, but got {type(lora_root_model).__name__}."
            )
        lora_root_model.token_initializer = inject_lora_into_model(
            lora_root_model.token_initializer,
            target_keywords=cfg.lora.target_keywords,
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            freeze_all=True,
        )

    if hasattr(model, "model"):
        model.model = lora_root_model
    else:
        trainer.state["model"] = lora_root_model

    if hasattr(model, "shadow"):
        model.shadow = deepcopy(lora_root_model)

    trainer.state["model"] = model

    trainable, total = count_trainable_parameters(lora_root_model)
    ranked_logger.info(
        f"LoRA enabled: trainable params={trainable:,} / total params={total:,} "
        f"({100.0 * trainable / total:.4f}%)"
    )
    ranked_logger.info("LoRA trainable parameter summary")


def _apply_nmf_if_enabled(cfg: DictConfig, trainer) -> None:
    if not (cfg.get("nmf", None) and cfg.nmf.enabled):
        return

    inject_nmf_into_model, count_trainable_parameters = _import_nmf_utils()
    model = trainer.state["model"]
    base_model = _unwrap_model(model)

    ranked_logger = logging.getLogger(__name__)
    ranked_logger.info(f"Applying replacement NMF to base model type: {type(base_model).__name__}")

    nmf_root_model = deepcopy(base_model)
    if not hasattr(nmf_root_model, "diffusion_module"):
        raise AttributeError(
            f"Expected the base model to expose `diffusion_module`, but got {type(nmf_root_model).__name__}."
        )

    apply_to_token_initializer = bool(cfg.nmf.get("apply_to_token_initializer", False))
    for p in nmf_root_model.parameters():
        p.requires_grad = False
    inject_kwargs = dict(
        target_keywords=cfg.nmf.target_keywords,
        rank=cfg.nmf.rank,
        nmf_alpha=cfg.nmf.alpha,
        nmf_eps=cfg.nmf.eps,
        freeze_all=False,
    )
    if apply_to_token_initializer:
        if not hasattr(nmf_root_model, "token_initializer"):
            raise AttributeError(
                f"Expected the base model to expose `token_initializer`, but got {type(nmf_root_model).__name__}."
            )
        nmf_root_model, nmf_records = inject_nmf_into_model(nmf_root_model, **inject_kwargs)
    else:
        nmf_root_model.diffusion_module, nmf_records = inject_nmf_into_model(
            nmf_root_model.diffusion_module,
            **inject_kwargs,
        )

    if hasattr(model, "model"):
        model.model = nmf_root_model
    else:
        trainer.state["model"] = nmf_root_model

    if hasattr(model, "shadow"):
        model.shadow = deepcopy(nmf_root_model)

    trainer.state["model"] = model

    trainable, total = count_trainable_parameters(nmf_root_model)
    ranked_logger.info(
        f"NMF enabled: trainable params={trainable:,} / total params={total:,} "
        f"({100.0 * trainable / total:.4f}%)"
    )
    ranked_logger.info("NMF replacement summary:")
    for rec in nmf_records:
        ranked_logger.info(
            f"  - {rec.module_name} [{rec.module_type}] in={rec.in_features} out={rec.out_features} "
            f"rank={rec.rank} base_params={rec.base_params:,} nmf_params={rec.nmf_params:,} "
            f"delta={rec.reduction:+,}"
        )

    nmf_dump = {
        "enabled": True,
        "apply_to_token_initializer": apply_to_token_initializer,
        "target_keywords": list(cfg.nmf.target_keywords),
        "rank": int(cfg.nmf.rank),
        "alpha": float(cfg.nmf.alpha),
        "eps": float(cfg.nmf.eps),
        "records": [
            {
                "module_name": rec.module_name,
                "module_type": rec.module_type,
                "in_features": rec.in_features,
                "out_features": rec.out_features,
                "rank": rec.rank,
                "base_params": rec.base_params,
                "nmf_params": rec.nmf_params,
                "reduction": rec.reduction,
            }
            for rec in nmf_records
        ],
    }
    sweep_dir = Path(cfg.paths.log_dir)
    dump_path = sweep_dir / f"{getattr(cfg, 'name', 'nmf')}.nmf_replacements.json"
    dump_path.write_text(json.dumps(nmf_dump, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ranked_logger.info(f"Wrote NMF replacement JSON to {dump_path}")
    _log_trainable_parameters(nmf_root_model, title="NMF trainable parameter summary")


def _build_weight_loading_config(raw_cfg) -> WeightLoadingConfig | None:
    if raw_cfg is None:
        return None
    if isinstance(raw_cfg, WeightLoadingConfig):
        return raw_cfg
    if OmegaConf.is_config(raw_cfg):
        raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    if isinstance(raw_cfg, dict):
        return WeightLoadingConfig(**raw_cfg)
    return hydra.utils.instantiate(raw_cfg)


def _build_parameter_freezing_config(raw_cfg) -> ParameterFreezingConfig | None:
    if raw_cfg is None:
        return None
    if isinstance(raw_cfg, ParameterFreezingConfig):
        return raw_cfg
    if OmegaConf.is_config(raw_cfg):
        raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    if isinstance(raw_cfg, dict):
        return ParameterFreezingConfig(**raw_cfg)
    return hydra.utils.instantiate(raw_cfg)


def _log_trainable_parameters(model, title: str = "Trainable parameter summary") -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / total if total else 0.0

    ranked_logger = logging.getLogger(__name__)
    ranked_logger.info(f"{title}: trainable params={trainable:,} / total params={total:,} ({ratio:.4f}%)")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


def _get_run_summary_path(cfg: DictConfig, trainer) -> Path:
    log_dir = Path(cfg.paths.log_dir)
    run_name = str(getattr(cfg, "name", "run"))
    global_step = trainer.state.get("global_step", None)
    if global_step is not None:
        return log_dir / f"{run_name}.step{global_step}.run_summary.json"
    return log_dir / f"{run_name}.run_summary.json"


def _safe_jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _scan_metrics_csv(log_dir: Path) -> tuple[Path | None, list[dict[str, str]]]:
    cands = list(log_dir.glob("**/lightning_logs/**/metrics.csv"))
    if not cands:
        return None, []
    metrics_csv = max(cands, key=lambda p: p.stat().st_mtime)
    try:
        import csv

        with metrics_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        rows = []
    return metrics_csv, rows


def _last_float(rows: list[dict[str, str]], *keys: str):
    last = None
    for row in rows:
        for key in keys:
            v = row.get(key, "")
            if v in (None, ""):
                continue
            try:
                last = float(v)
                break
            except ValueError:
                continue
    return last


def _best_epoch_from_rows(rows: list[dict[str, str]]):
    scored = []
    for row in rows:
        epoch = row.get("epoch") or row.get("trainer/global_step") or row.get("step")
        try:
            epoch = int(float(epoch)) if epoch not in (None, "") else None
        except ValueError:
            epoch = None
        if epoch is None:
            continue
        loss = row.get("val/total_loss") or row.get("train/per_epoch_total_loss")
        lddt = row.get("val/mean_lddt") or row.get("train/per_epoch_mean_lddt_protein")
        try:
            loss_f = float(loss) if loss not in (None, "") else None
        except ValueError:
            loss_f = None
        try:
            lddt_f = float(lddt) if lddt not in (None, "") else None
        except ValueError:
            lddt_f = None
        if loss_f is not None or lddt_f is not None:
            scored.append((epoch, loss_f, lddt_f))
    if not scored:
        return None, None, None
    scored.sort(key=lambda x: (float('inf') if x[1] is None else x[1], float('-inf') if x[2] is None else -x[2], x[0]))
    return scored[0]


def _write_run_summary(cfg: DictConfig, trainer, extra: dict[str, Any]) -> Path:
    summary_path = _get_run_summary_path(cfg, trainer)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    model = _unwrap_model(trainer.state["model"])
    trainable, total = (None, None)
    try:
        trainable, total = (sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
    except Exception:
        pass

    current_epoch = trainer.state.get("current_epoch", None)
    global_step = trainer.state.get("global_step", None)

    metrics = extra.get("metrics", {})
    payload = {
        "name": str(getattr(cfg, "name", "run")),
        "run_dir": str(Path(cfg.paths.log_dir)),
        "current_epoch": current_epoch,
        "global_step": global_step,
        "trainable_params": trainable,
        "total_params": total,
        "status": extra.get("status", "unknown"),
        "best_epoch": extra.get("best_epoch"),
        "best_epoch_loss": extra.get("best_epoch_loss"),
        "best_epoch_lddt": extra.get("best_epoch_lddt"),
        "final_epoch": extra.get("final_epoch", current_epoch),
        "final_epoch_loss": extra.get("final_epoch_loss"),
        "final_epoch_lddt": extra.get("final_epoch_lddt"),
        "best_val_lddt": metrics.get("best_val_lddt", extra.get("best_val_lddt")),
        "best_val_loss": metrics.get("best_val_loss", extra.get("best_val_loss")),
        "final_val_lddt": metrics.get("final_val_lddt", extra.get("final_val_lddt")),
        "final_val_loss": metrics.get("final_val_loss", extra.get("final_val_loss")),
        "metrics": _safe_jsonable(metrics),
        "nmf": _safe_jsonable(extra.get("nmf", {})),
        "lora": _safe_jsonable(extra.get("lora", {})),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def _build_ckpt_config(cfg: DictConfig) -> CheckpointConfig | None:
    if "ckpt_config" in cfg and cfg.ckpt_config:
        ckpt_cfg = cfg.ckpt_config
        if OmegaConf.is_config(ckpt_cfg):
            ckpt_cfg = OmegaConf.to_container(ckpt_cfg, resolve=True)
        return CheckpointConfig(
            path=ckpt_cfg["path"],
            reset_optimizer=ckpt_cfg.get("reset_optimizer", False),
            weight_loading_config=_build_weight_loading_config(
                ckpt_cfg.get("weight_loading_config", None)
            ),
            parameter_freezing_config=_build_parameter_freezing_config(
                ckpt_cfg.get("parameter_freezing_config", None)
            ),
        )

    if "ckpt_path" in cfg and cfg.ckpt_path is not None:
        return CheckpointConfig(path=cfg.ckpt_path)

    return None


def _wrap_checkpoint_loader_for_lora(cfg: DictConfig, trainer) -> None:
    if not (cfg.get("lora", None) and cfg.lora.enabled):
        return

    original_load_checkpoint = trainer.load_checkpoint

    def load_checkpoint_and_reset_progress(*args, **kwargs):
        original_load_checkpoint(*args, **kwargs)
        trainer.state["current_epoch"] = 0
        trainer.state["global_step"] = 0
        trainer.should_stop = False

        base_model = _unwrap_model(trainer.state["model"])
        if hasattr(base_model, "train"):
            base_model.train()

        logging.getLogger(__name__).info(
            "LoRA checkpoint loaded as weights-only start: reset current_epoch and global_step to 0."
        )

    trainer.load_checkpoint = load_checkpoint_and_reset_progress


@hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    _spawning_process_logger.info("Importing dependencies...")

    import torch
    from lightning.fabric import seed_everything
    from lightning.fabric.loggers import Logger

    torch.set_float32_matmul_precision("medium")

    from foundry.callbacks.callback import BaseCallback  # noqa
    from foundry.utils.instantiators import instantiate_loggers, instantiate_callbacks  # noqa
    from foundry.utils.logging import (
        print_config_tree,
        log_hyperparameters_with_all_loggers,
    )  # noqa
    from foundry.utils.ddp import RankedLogger  # noqa
    from foundry.utils.ddp import is_rank_zero, set_accelerator_based_on_availability  # noqa
    from foundry.utils.datasets import (
        recursively_instantiate_datasets_and_samplers,
        assemble_distributed_loader,
        subset_dataset_to_example_ids,
        assemble_val_loader_dict,
    )  # noqa

    set_accelerator_based_on_availability(cfg)
    ranked_logger = RankedLogger(__name__, rank_zero_only=True)
    _spawning_process_logger.info("Completed dependency imports ...")

    print_config_tree(cfg, resolve=True)

    if not is_rank_zero():
        dataset_logger = logging.getLogger("datasets")
        sampler_logger = logging.getLogger("atomworks.ml.samplers")
        dataset_logger.setLevel(logging.WARNING)
        sampler_logger.setLevel(logging.ERROR)

    if cfg.get("seed"):
        ranked_logger.info(f"Seeding everything with seed={cfg.seed}...")
        seed_everything(cfg.seed, workers=True, verbose=True)
    else:
        ranked_logger.warning("No seed provided - Not seeding anything!")

    ranked_logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    ranked_logger.info("Instantiating callbacks...")
    callbacks: list[BaseCallback] = instantiate_callbacks(cfg.get("callbacks"))

    ranked_logger.info("Instantiating trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=loggers or None,
        callbacks=callbacks or None,
        _convert_="partial",
        _recursive_=False,
    )
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})

    ranked_logger.info(
        f"Spawning {trainer.fabric.world_size} processes from {trainer.fabric.global_rank}..."
    )
    trainer.fabric.launch()

    ranked_logger.info("Constructing model...")
    trainer.construct_model()

    _apply_lora_if_enabled(cfg, trainer)
    _apply_nmf_if_enabled(cfg, trainer)

    trainer.construct_optimizer()
    trainer.construct_scheduler()

    n_examples_per_epoch = cfg.trainer.n_examples_per_epoch

    assert (
        "train" in cfg.datasets and cfg.datasets.train
    ), "No 'train' dataloader configuration provided! If only performing validation, use `validate.py` instead."
    dataset_and_sampler = recursively_instantiate_datasets_and_samplers(
        cfg.datasets.train
    )
    train_dataset, train_sampler = (
        dataset_and_sampler["dataset"],
        dataset_and_sampler["sampler"],
    )

    if "subset_to_example_ids" in cfg.datasets:
        train_dataset = subset_dataset_to_example_ids(
            train_dataset, cfg.datasets.subset_to_example_ids
        )
        train_sampler = None

    train_loader = assemble_distributed_loader(
        dataset=train_dataset,
        sampler=train_sampler,
        rank=trainer.fabric.global_rank,
        world_size=trainer.fabric.world_size,
        n_examples_per_epoch=n_examples_per_epoch,
        loader_cfg=cfg.dataloader["train"],
    )

    if "val" in cfg.datasets and cfg.datasets.val:
        val_loaders = assemble_val_loader_dict(
            cfg=cfg.datasets.val,
            rank=trainer.fabric.global_rank,
            world_size=trainer.fabric.world_size,
            loader_cfg=cfg.dataloader["val"],
        )
    else:
        ranked_logger.warning("No validation datasets provided! Skipping validation...")
        val_loaders = None

    ranked_logger.info("Logging hyperparameters...")
    log_hyperparameters_with_all_loggers(
        trainer=trainer, cfg=cfg, model=trainer.state["model"]
    )

    ckpt_config = _build_ckpt_config(cfg)
    _wrap_checkpoint_loader_for_lora(cfg, trainer)

    if ckpt_config is not None and ckpt_config.parameter_freezing_config is not None:
        _log_trainable_parameters(
            _unwrap_model(trainer.state["model"]),
            title="Post-freeze trainable parameter summary",
        )

    ranked_logger.info("Training model...")

    final_status = "unknown"
    summary_payload: dict[str, Any] = {
        "metrics": {},
        "nmf": {},
        "lora": {},
        "status": "unknown",
    }

    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v.item() if hasattr(v, "item") else v)
        except Exception:
            return None

    def _collect_metrics() -> dict[str, Any]:
        sources = []
        if hasattr(trainer, "state"):
            sources.append(trainer.state.get("metrics", None))
        sources.append(getattr(trainer, "callback_metrics", None))
        model = _unwrap_model(trainer.state["model"])
        sources.append(getattr(model, "callback_metrics", None))
        metrics: dict[str, Any] = {}
        for source in sources:
            if not source:
                continue
            try:
                if hasattr(source, "items"):
                    for k, v in source.items():
                        metrics[str(k)] = _to_float(v)
            except Exception:
                continue
        return metrics

    try:
        with suppress_warnings():
            trainer.fit(
                train_loader=train_loader, val_loaders=val_loaders, ckpt_config=ckpt_config
            )
        final_status = "success"
    except Exception:
        final_status = "error"
        raise
    finally:
        current_epoch = trainer.state.get("current_epoch", None)
        metrics = _collect_metrics()
        metrics_csv, csv_rows = _scan_metrics_csv(Path(cfg.paths.log_dir))
        if metrics_csv is not None:
            summary_payload["metrics_csv"] = str(metrics_csv)
            csv_best_lddt = _last_float(csv_rows, "val/mean_lddt", "train/per_epoch_mean_lddt_protein")
            csv_best_loss = _last_float(csv_rows, "val/total_loss", "train/per_epoch_total_loss")
            csv_best_epoch, csv_best_epoch_loss, csv_best_epoch_lddt = _best_epoch_from_rows(csv_rows)
            final_row = csv_rows[-1] if csv_rows else {}
            csv_final_epoch = None
            try:
                csv_final_epoch = int(float(final_row.get("epoch", final_row.get("step", current_epoch)))) if final_row else current_epoch
            except Exception:
                csv_final_epoch = current_epoch
            csv_final_loss = _last_float([final_row], "val/total_loss", "train/per_epoch_total_loss") if final_row else None
            csv_final_lddt = _last_float([final_row], "val/mean_lddt", "train/per_epoch_mean_lddt_protein") if final_row else None
            if csv_best_lddt is not None:
                metrics["val/mean_lddt"] = csv_best_lddt
            if csv_best_loss is not None:
                metrics["val/total_loss"] = csv_best_loss
            summary_payload["best_epoch"] = csv_best_epoch
            summary_payload["best_epoch_loss"] = csv_best_epoch_loss
            summary_payload["best_epoch_lddt"] = csv_best_epoch_lddt
            summary_payload["final_epoch"] = csv_final_epoch
            summary_payload["final_epoch_loss"] = csv_final_loss
            summary_payload["final_epoch_lddt"] = csv_final_lddt
        summary_payload["metrics"] = metrics
        summary_payload["best_val_lddt"] = metrics.get("val/mean_lddt") or metrics.get("train/per_epoch_mean_lddt_protein")
        summary_payload["best_val_loss"] = metrics.get("val/total_loss") or metrics.get("train/per_epoch_total_loss")
        summary_payload["final_val_lddt"] = summary_payload["best_val_lddt"]
        summary_payload["final_val_loss"] = summary_payload["best_val_loss"]
        summary_payload.setdefault("best_epoch", current_epoch)
        summary_payload.setdefault("best_epoch_loss", summary_payload["best_val_loss"])
        summary_payload.setdefault("best_epoch_lddt", summary_payload["best_val_lddt"])
        summary_payload.setdefault("final_epoch", current_epoch)
        summary_payload.setdefault("final_epoch_loss", summary_payload["final_val_loss"])
        summary_payload.setdefault("final_epoch_lddt", summary_payload["final_val_lddt"])
        if cfg.get("nmf", None) and cfg.nmf.enabled:
            summary_payload["nmf"] = {
                "enabled": True,
                "apply_to_token_initializer": bool(cfg.nmf.get("apply_to_token_initializer", False)),
                "target_keywords": list(cfg.nmf.target_keywords),
                "rank": int(cfg.nmf.rank),
                "alpha": float(cfg.nmf.alpha),
                "eps": float(cfg.nmf.eps),
            }
        if cfg.get("lora", None) and cfg.lora.enabled:
            summary_payload["lora"] = {
                "enabled": True,
                "target_keywords": list(cfg.lora.target_keywords),
                "rank": int(cfg.lora.rank),
                "alpha": float(cfg.lora.alpha),
                "dropout": float(cfg.lora.dropout),
            }
        summary_payload["status"] = final_status
        try:
            summary_path = _write_run_summary(cfg, trainer, summary_payload)
            ranked_logger.info(f"Wrote run summary JSON to {summary_path}")
        except Exception as summary_exc:
            ranked_logger.warning(f"Failed to write run summary JSON: {summary_exc}")


if __name__ == "__main__":
    train()
