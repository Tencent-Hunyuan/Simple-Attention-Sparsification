# this file is adapted from VeOmni trainer
# https://github.com/ByteDance-Seed/VeOmni/blob/main/tasks/train_torch.py
import json
import math
import os
import time
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

# Disable liger kernel if requested
if os.environ.get("USE_LIGER_KERNEL", "1") == "0":
    try:
        import veomni.utils.import_utils
        veomni.utils.import_utils._PACKAGE_FLAGS["liger_kernel"] = False
    except ImportError:
        pass

from veomni.arguments import parse_args, save_args

from veomni.checkpoint import build_checkpointer
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets
from veomni.optim import build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.loss_utils import count_loss_token, mean_global_loss

from .config import SasArguments


logger = helper.create_logger(__name__)


def freeze_all_but_router(model):
    for name, param in model.named_parameters():
        param.requires_grad = "router" in name

    logger.info_rank0("Freeze all parameters except router.")


def gather_router_state_dict(model):
    """Gather the (possibly FSDP2-sharded) router params to a full CPU state dict.

    ``DTensor.full_tensor()`` is a collective, so every rank must call it; only
    rank 0 keeps the result. This lets us export AttnGates straight from the live
    model without first writing a full-model checkpoint.
    """
    from torch.distributed.tensor import DTensor

    is_rank0 = dist.get_rank() == 0
    state = {}
    for name, param in model.named_parameters():
        if "router" not in name:
            continue
        t = param.full_tensor() if isinstance(param, DTensor) else param
        if is_rank0:
            state[name] = t.detach().cpu()
    return state


def main():
    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    logger.info(f"Process_group timeout: {nccl_timeout}")
    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = parse_args(SasArguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)
    
    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader_type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        dyn_bsz=args.train.dyn_bsz,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        pad_packed_to_length=args.train.pad_packed_to_length,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    model_config = model.config

    helper.print_device_mem_info("VRAM usage after building model")

    ####################
    if args.sparse.sparse_mod:
        from sas.core.patch import hf_convert
        # Save sparse attention metadata to config
        model_config.sas_sparse_mod = args.sparse.sparse_mod
        model_config.sas_sparse_config = args.sparse.to_sparse_config()
        
        model = hf_convert(
            model,
            sparse_mod=args.sparse.sparse_mod,
            sparse_config=args.sparse.to_sparse_config(),
        )

    # Router-only training: freeze everything except the router.
    freeze_all_but_router(model)

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_reshard_after_forward=args.train.enable_reshard_after_forward,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    # Re-initialize router weights after parallelization (aligned with SeerAttn).
    # HF's meta-init / FSDP2 sharding can override the router's constructor init,
    # so we re-enforce it here via each router module's reset_sparse_parameters().
    from sas.core.attn import init_router_weights
    init_router_weights(model)

    logger.info_rank0("Collecting router parameters for the optimizer...")

    router_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and "router" in name
    ]

    param_groups = [{
        "params": router_params,
        "weight_decay": args.sparse.router_weight_decay,
        "lr": args.sparse.router_lr,
    }]

    logger.info_rank0(f"Router-only mode: Router LR: {args.sparse.router_lr} (LLM frozen)")
    logger.info_rank0(
        f"Router schedule: decay={args.sparse.router_lr_decay_style}, warmup={args.sparse.router_lr_warmup_ratio}, min_lr={args.sparse.router_lr_min}"
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        param_groups=param_groups,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    # --- Build per-group LambdaLR scheduler ---
    # Helper to create a schedule lambda given LR config
    def _make_schedule_lambda(init_lr, decay_style, warmup_ratio, lr_min, lr_start=0.0, lr_decay_ratio=1.0):
        total_steps = args.train.train_steps * args.train.num_train_epochs
        num_warmup_steps = int(total_steps * warmup_ratio)

        if decay_style == "constant":
            def _lambda(step):
                if step < num_warmup_steps:
                    return (lr_start + (init_lr - lr_start) * step / max(1, num_warmup_steps)) / init_lr
                return 1.0
            return _lambda

        elif decay_style == "linear":
            min_lr_ratio = lr_min / init_lr
            def _lambda(step):
                if step < num_warmup_steps:
                    return (lr_start + (init_lr - lr_start) * step / max(1, num_warmup_steps)) / init_lr
                return max(
                    min_lr_ratio,
                    float(total_steps - step) / float(max(1, total_steps - num_warmup_steps)),
                )
            return _lambda

        elif decay_style == "cosine":
            lr_decay_steps = int(total_steps * lr_decay_ratio)
            min_lr_ratio = lr_min / init_lr
            def _lambda(step):
                if step < num_warmup_steps:
                    return (lr_start + (init_lr - lr_start) * step / max(1, num_warmup_steps)) / init_lr
                if step > lr_decay_steps:
                    return min_lr_ratio
                progress = float(step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                factor = factor * (1 - min_lr_ratio) + min_lr_ratio
                return max(0, factor)
            return _lambda

        else:
            raise ValueError(f"Unknown lr_decay_style: {decay_style}")

    # Build the router LR schedule (one param group).
    lr_lambdas = [_make_schedule_lambda(
        init_lr=args.sparse.router_lr,
        decay_style=args.sparse.router_lr_decay_style,
        warmup_ratio=args.sparse.router_lr_warmup_ratio,
        lr_min=args.sparse.router_lr_min,
    )]

    from torch.optim.lr_scheduler import LambdaLR
    lr_scheduler = LambdaLR(optimizer, lr_lambdas)

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=wandb.Settings(console="off"),
                config={
                    **vars(args.model),
                    **vars(args.data),
                    **vars(args.train),
                    **vars(args.sparse),
                },  # flatten dict
            )

        # save model_assets before training
        model_assets = [model_config, tokenizer if args.data.data_type == "plaintext" else chat_template]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        # Save a snapshot of optimizer param_group hyperparameters BEFORE DCP load.
        # DCP's _split_optim_state_dict may fail to match param_groups (e.g. when
        # model structure or frozen params differ from the checkpoint), resulting in
        # param_groups that contain ONLY the 'params' key — all hyperparameters
        # (lr, betas, eps, weight_decay, ...) are silently lost.
        _pg_snapshots = [
            {k: v for k, v in pg.items() if k != "params"}
            for pg in optimizer.param_groups
        ]

        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)

        # Restore any optimizer param_group keys that DCP load silently dropped.
        for pg, snapshot in zip(optimizer.param_groups, _pg_snapshots):
            for key, val in snapshot.items():
                if key not in pg:
                    pg[key] = val
                    logger.info_rank0(f"Repaired missing optimizer param_group key: {key}={val}")

        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            micro_batches_token_num = count_loss_token(micro_batches)
            num_micro_steps = len(micro_batches)

            for micro_step, micro_batch in enumerate(micro_batches):
                if (
                    args.train.data_parallel_mode == "fsdp2"
                    and not args.train.enable_reshard_after_backward
                    and num_micro_steps > 1
                ):
                    if micro_step == 0:
                        model.set_reshard_after_backward(False)
                    elif micro_step == num_micro_steps - 1:
                        model.set_reshard_after_backward(True)
                environ_meter.add(micro_batch)
                micro_batch_token_num = count_loss_token(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    loss = model(**micro_batch, use_cache=False).loss

                loss, _ = mean_global_loss(loss, micro_batch_token_num, micro_batches_token_num)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            
            # Router-only LR logging (single param group -> single LR)
            last_lrs = lr_scheduler.get_last_lr()
            lr_router = max(last_lrs) if isinstance(last_lrs, list) else last_lrs

            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: router:{lr_router:.2e}", refresh=False
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update({
                        "training/loss": total_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr_router": lr_router,
                    })
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if bool(args.train.save_steps) and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # Export the trained gate router as an sglang AttnGates dir. Training is
    # router-only (backbone frozen), so we gather just the small router state
    # straight from the live model — no full-model checkpoint needed. The gather
    # is a collective, so ALL ranks must call it; only rank 0 writes the dir.
    if args.train.save_hf_weights:
        router_state_dict = gather_router_state_dict(model)
        if args.train.global_rank == 0:
            from sas.train.export_gate import export_attn_gates
            gates_path = os.path.join(args.train.save_checkpoint_path, "AttnGates")
            export_attn_gates(
                state_dict=router_state_dict,
                model_config=model_config,
                base_model=args.model.model_path,
                output_dir=gates_path,
                tokenizer_dir=args.train.model_assets_dir,
            )
            logger.info_rank0(f"AttnGates exported to {gates_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
