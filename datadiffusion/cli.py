"""datadiffusion — train a tabular diffusion model, sample from it, and evaluate it against baselines.

Each subcommand imports its heavy modules lazily so `--help` stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datadiffusion import __version__

logger = logging.getLogger(__name__)


def _config(args: argparse.Namespace):
    from dataclasses import replace

    from datadiffusion.config import ExperimentConfig

    cfg = ExperimentConfig.from_toml(args.config) if args.config else ExperimentConfig()
    # explicit CLI flags win over the config file
    if args.name:
        cfg.name = args.name
    if args.dataset:
        cfg.data = replace(cfg.data, dataset=args.dataset)
    if args.epochs is not None:
        cfg.train = replace(cfg.train, epochs=args.epochs)
    if args.seed is not None:
        cfg.train = replace(cfg.train, seed=args.seed)
        cfg.data = replace(cfg.data, seed=args.seed)
    return cfg


def _data(cfg):
    from dataclasses import asdict

    from datadiffusion.data import load_dataset

    d = asdict(cfg.data)
    return load_dataset(d.pop("dataset"), **d)


def _cmd_train(args: argparse.Namespace) -> int:
    import torch

    from datadiffusion.runs import new_run_dir
    from datadiffusion.train import train

    torch.set_num_threads(args.threads)
    cfg = _config(args)
    run = new_run_dir(args.out, cfg.name if args.name or args.config else cfg.data.dataset)
    (run / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    summary = train(cfg, _data(cfg), run)
    logger.info("run: %s  %s", run, summary)
    print(run)
    return 0


def _generate(run: Path, n: int | None, method: str, steps: int, eta: float, clip: float | None, seed: int):
    from datadiffusion.data import Preprocessor
    from datadiffusion.runs import load_run
    from datadiffusion.sampling import sample

    cfg, model, schedule = load_run(run)
    data = _data(cfg)
    pre = Preprocessor(cfg.data.seed).fit(data.train)
    z = sample(
        model,
        schedule,
        n or len(data.holdout),
        data.dim,
        method=method,
        steps=steps,
        eta=eta,
        clip=clip or None,
        seed=seed,
    )
    return data, pre.inverse_transform(z.numpy())


def _cmd_sample(args: argparse.Namespace) -> int:
    from datadiffusion.runs import save_table

    data, rows = _generate(Path(args.run), args.n, args.method, args.steps, args.eta, args.clip, args.seed)
    path = save_table(Path(args.out or Path(args.run) / "samples.csv"), rows, data.columns)
    logger.info("wrote %d rows to %s", len(rows), path)
    print(path)
    return 0


def _write_report(rep: dict, data, synthetic: dict, out: Path) -> None:
    from datadiffusion.evaluate import to_markdown

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "report.md").write_text(to_markdown(rep))
    try:
        from datadiffusion import plots
    except ImportError:  # pragma: no cover - plots extra missing
        return
    try:
        plots.marginals(data.holdout, synthetic, data.columns, out / "marginals.png")
        plots.correlations(data.holdout, synthetic, data.columns, out / "correlations.png")
        if data.dim == 2:
            plots.scatter2d(data.holdout, synthetic, out / "scatter.png")
    except ImportError:
        logger.info("install the [plots] extra for figures")


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from datadiffusion.evaluate import BASELINES, report, to_markdown
    from datadiffusion.runs import load_table

    run = Path(args.run)
    if args.samples:
        from datadiffusion.runs import load_run

        cfg, _, _ = load_run(run)
        data, rows = _data(cfg), load_table(Path(args.samples))
    else:
        data, rows = _generate(run, None, args.method, args.steps, args.eta, args.clip, args.seed)
    rep = report({"diffusion": rows}, data, seed=args.seed)
    synthetic = {
        "diffusion": rows,
        **{k: f(data.train, len(data.holdout), args.seed) for k, f in BASELINES.items()},
    }
    _write_report(rep, data, synthetic, run)
    print(to_markdown(rep))
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    from datadiffusion.data import load_dataset
    from datadiffusion.evaluate import report, to_markdown

    data = load_dataset(args.dataset, seed=args.seed)
    print(to_markdown(report({}, data, seed=args.seed)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datadiffusion", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train a model; prints the run directory")
    p.add_argument("--config", help="TOML with [data] [model] [diffusion] [train] sections")
    p.add_argument("--dataset", choices=["california", "moons", "circles"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="runs")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=_cmd_train)

    def add_sampling(p):
        p.add_argument("--method", choices=["ddim", "ddpm"], default="ddim")
        p.add_argument("--steps", type=int, default=50, help="DDIM steps")
        p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 deterministic, 1 ~DDPM)")
        p.add_argument("--clip", type=float, default=5.0, help="clip predicted x0 to [-clip, clip] (0 = off)")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("sample", help="generate synthetic rows (CSV, original units)")
    p.add_argument("run")
    p.add_argument("-n", "--n", type=int, default=None, help="rows (default: size of the holdout split)")
    p.add_argument("-o", "--out", default=None, help="CSV path (default RUN/samples.csv)")
    add_sampling(p)
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser("evaluate", help="compare samples with held-out data and baselines")
    p.add_argument("run")
    p.add_argument("--samples", default=None, help="CSV to evaluate instead of sampling fresh")
    add_sampling(p)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("baselines", help="evaluate only the baselines on a dataset")
    p.add_argument("--dataset", choices=["california", "moons", "circles"], default="california")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_baselines)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return args.func(args)
