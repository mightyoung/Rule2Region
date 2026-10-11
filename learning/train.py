"""Single-device local head trainer/profile. No weights download or paid service."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from .contract import fingerprint, require


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=5, help="total steps including resumed steps")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--software-test", action="store_true")
    return parser.parse_args()


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from .dataset import RelationDataset
    from .model import RelationHead
    require(args.steps > 0 and args.lr > 0 and args.hidden > 0, "training settings")
    manifest = json.loads(args.manifest.read_text())
    cache = json.loads(args.features.read_text())
    dataset = RelationDataset(manifest, cache, "train", args.software_test)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    model = RelationHead(cache["dimension"], args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {"manifest_sha256": fingerprint(manifest), "features_sha256": fingerprint(cache), "seed": args.seed, "dimension": cache["dimension"], "hidden": args.hidden, "lr": args.lr, "purpose": manifest["purpose"], "device": args.device, "torch_version": str(torch.__version__), "trainer_version": "boundary-head-v1"}
    start = 0
    require(args.resume or not args.checkpoint.exists(), "checkpoint exists: resume or choose new path")
    if args.resume:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        require(state["config"] == config, "resume config/manifest mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    require(start < args.steps, "checkpoint already at requested total steps")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    gpu = args.device.startswith("cuda")
    if gpu:
        torch.cuda.reset_peak_memory_stats(args.device)
    profile = []
    model.train()
    for step in range(start, args.steps):
        if gpu:
            torch.cuda.synchronize(args.device)
        begin = time.perf_counter()
        inputs, targets = dataset[step % len(dataset)]
        outputs = model(**{k: v.to(args.device) for k, v in inputs.items()})
        losses = [torch.nn.functional.cross_entropy(value, targets[key].to(args.device)) for key, value in outputs.items() if len(targets[key])]
        require(losses, "no candidate targets")
        loss = torch.stack(losses).mean()
        require(torch.isfinite(loss).item(), "non-finite loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if gpu:
            torch.cuda.synchronize(args.device)
        profile.append({"step": step + 1, "loss": loss.item(), "step_seconds": time.perf_counter() - begin})
        state = {"config": config, "step": step + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if gpu else []}
        temporary = args.checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(args.checkpoint)
    return {"kind": "software_test_only" if args.software_test else "local_head_profile", "config": config, "peak_cuda_bytes": torch.cuda.max_memory_allocated(args.device) if gpu else None, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if gpu else None, "profile": profile}


def main():
    args = arguments()
    try:
        print(json.dumps(run(args), allow_nan=False))
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("PyTorch unavailable: training/gradient/profile NOT RUN; no dependency installation") from exc
        raise


if __name__ == "__main__":
    main()
