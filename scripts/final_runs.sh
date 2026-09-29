#!/bin/bash
# Reproduce the README results (v3.0.0): train each config, then evaluate with DDIM (saved as ddim_*) and DDPM (report.*)
set -e
cd "$(dirname "$0")/.."
for cfg in moons california; do
  run=$(uv run datadiffusion train --config configs/$cfg.toml --threads 2 | tail -1)
  echo "RUN $cfg $run"
  uv run datadiffusion evaluate "$run" --method ddim --steps 100
  for f in report.json report.md marginals.png correlations.png scatter.png; do
    [ -f "$run/$f" ] && mv "$run/$f" "$run/ddim_$f"
  done
  uv run datadiffusion evaluate "$run" --method ddpm
done
echo ALL_DONE
