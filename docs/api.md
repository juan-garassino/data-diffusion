# API Reference

## Configuration

### `datadiffusion.config`

All configuration is managed through nested dataclasses. The top-level `ExperimentConfig` can be constructed from CLI args, YAML, or directly in code.

```python
from datadiffusion.config import ExperimentConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    model=ModelConfig(hidden_size=256, num_layers=4),
    training=TrainingConfig(num_epochs=100),
)
```

#### `DataConfig`
| Field | Type | Default | Description |
|---|---|---|---|
| `scaler_type` | `str` | `"quantile"` | `"quantile"` or `"standard"` |

#### `ModelConfig`
| Field | Type | Default | Description |
|---|---|---|---|
| `model_type` | `str` | `"mlp"` | `"mlp"`, `"transformer"`, or `"score"` |
| `hidden_size` | `int` | `256` | Hidden layer width |
| `num_layers` | `int` | `4` | Number of residual blocks / transformer layers |
| `num_heads` | `int` | `4` | Attention heads (transformer only) |
| `dropout` | `float` | `0.1` | Dropout rate |
| `embedding_type` | `str` | `"sinusoidal"` | Time embedding type |
| `embedding_kwargs` | `dict` | `{}` | Extra kwargs for embedding |

#### `SchedulerConfig`
| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler_type` | `str` | `"standard"` | `"standard"` (DDPM) or `"score"` (SDE) |
| `num_timesteps` | `int` | `1000` | Total diffusion timesteps |
| `beta_start` | `float` | `0.0001` | Starting noise level |
| `beta_end` | `float` | `0.02` | Ending noise level |
| `beta_schedule` | `str` | `"cosine"` | `"linear"`, `"quadratic"`, or `"cosine"` |

#### `TrainingConfig`
| Field | Type | Default | Description |
|---|---|---|---|
| `num_epochs` | `int` | `200` | Maximum training epochs |
| `learning_rate` | `float` | `3e-4` | Peak LR for OneCycleLR |
| `batch_size` | `int` | `64` | Training batch size |
| `accumulation_steps` | `int` | `4` | Gradient accumulation steps |
| `early_stopping_patience` | `int` | `25` | Patience for early stopping |
| `validation_split` | `float` | `0.1` | Validation fraction |
| `importance_sampling` | `bool` | `True` | Weight timesteps toward low t |

#### `GenerationConfig`
| Field | Type | Default | Description |
|---|---|---|---|
| `num_samples` | `int` | `1000` | Samples to generate |
| `sampler_type` | `str` | `"ddpm"` | `"ddpm"` or `"ddim"` |
| `ddim_steps` | `int` | `50` | DDIM inference steps |

#### `SelfImprovementConfig`
| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Enable self-improvement loop |
| `max_iterations` | `int` | `10` | Maximum loop iterations |
| `quality_threshold` | `float` | `0.80` | Target composite score |
| `convergence_delta` | `float` | `0.01` | Minimum improvement to avoid convergence |
| `lr_decay_factor` | `float` | `0.5` | LR multiplier on high KS |
| `epoch_increase_factor` | `float` | `1.5` | Epoch multiplier on poor quality |
| `capacity_increase_factor` | `float` | `1.5` | Hidden size multiplier on high corr diff |

#### `ExperimentConfig`

```python
config = ExperimentConfig(
    experiment_name="my_run",
    dataset="california",
    verbose=False,
    data=DataConfig(...),
    model=ModelConfig(...),
    scheduler=SchedulerConfig(...),
    training=TrainingConfig(...),
    generation=GenerationConfig(...),
    self_improvement=SelfImprovementConfig(...),
)

# Serialize
config.to_dict()  # → nested dict

# Load from YAML
config = ExperimentConfig.from_yaml("config.yaml")
```

---

## Data

### `datadiffusion.data.loading`

```python
def california_housing_dataset(scaler_type: str = "quantile") -> tuple[TensorDataset, transformer]
```

Loads California Housing (20640 samples, 8 features) and applies the specified scaler. Returns the scaled dataset and the fitted transformer for inverse-transforming generated samples.

```python
def get_dataset(name: str, scaler_type: str = "quantile") -> tuple[TensorDataset, transformer]
```

Dataset dispatcher. Currently supports `"california"`.

**Feature names** (in order):
```python
CALIFORNIA_FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
```

### `datadiffusion.data.utils`

```python
def normalize_data(data) -> tuple[normalized, mean, std]
def denormalize_data(normalized_data, mean, std) -> original
```

Simple mean/std normalization. Works with both numpy arrays and torch tensors.

---

## Models

### `datadiffusion.models.mlp.TabularMLP`

```python
model = TabularMLP(
    input_size=8,
    hidden_size=256,
    hidden_layers=4,
    dropout=0.1,
    embedding_type="sinusoidal",
)
output = model(x, t)  # (B, 8), (B,) → (B, 8)
```

Residual MLP with sinusoidal time embedding. Each hidden layer is a `ResidualBlock` with LayerNorm, two linear layers, GELU, dropout, and a skip connection.

### `datadiffusion.models.transformer.TabularTransformer`

```python
model = TabularTransformer(
    input_size=8,
    hidden_size=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    embedding_type="sinusoidal",
)
```

Treats each feature as a token. Uses `nn.TransformerEncoder` with the time embedding concatenated to each token.

### `datadiffusion.models.factory.create_model_and_scheduler`

```python
model, scheduler = create_model_and_scheduler(config, input_size=8)
```

Creates the appropriate model and scheduler based on `config.model.model_type` and `config.scheduler.scheduler_type`.

---

## Schedulers

### `datadiffusion.schedulers.NoiseScheduler`

```python
scheduler = NoiseScheduler(
    num_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="cosine",  # "linear", "quadratic", "cosine"
)

# Add noise to clean data
x_t = scheduler.add_noise(x_start, noise, timesteps)

# Reverse step
x_prev = scheduler.step(model_output, timestep, x_t)

# Precomputed tensors
scheduler.alphas_cumprod  # (T,)
scheduler.betas  # (T,)
len(scheduler)  # T
```

### `datadiffusion.schedulers.ScoreBasedNoiseScheduler`

Extends `NoiseScheduler` with a score-based SDE reverse step. Used with `model_type="score"`.

---

## Training

### `datadiffusion.training.trainer.Trainer`

```python
trainer = Trainer(config)
train_losses, val_losses = trainer.train(model, scheduler, dataset)
```

Full training loop with:
- Gradient accumulation
- OneCycleLR scheduling
- EMA weight averaging
- Early stopping on EMA validation loss
- Importance-weighted timestep sampling
- NaN/Inf detection and recovery

After training, the model holds the best EMA weights.

### `datadiffusion.training.trainer.EMA`

```python
ema = EMA(model, decay=0.999)
ema.update(model)  # Update shadow weights
ema.apply(model)  # Copy EMA weights into model
ema.restore(model)  # Restore original weights
```

### `datadiffusion.training.losses`

```python
loss = noise_prediction_loss(predicted, noise)  # MSE
loss = score_matching_loss(predicted, noise, sqrt_1m_alpha, timesteps)  # Weighted MSE
loss_fn = get_loss_fn("mlp")  # Returns noise_prediction_loss
```

---

## Generation

### `datadiffusion.generation.sampler`

```python
# DDPM (stochastic, T steps)
samples = generate_samples(model, scheduler, num_samples=1000, input_size=8)

# DDIM (deterministic, fewer steps)
samples = generate_samples_ddim(
    model,
    scheduler,
    num_samples=1000,
    input_size=8,
    num_inference_steps=50,
    eta=0.0,  # 0 = deterministic, 1 = equivalent to DDPM
)
```

Both return `np.ndarray` of shape `(num_samples, input_size)` in scaled space. Use `transformer.inverse_transform(samples)` to get original-space values.

---

## Evaluation

### `datadiffusion.evaluation.composite`

```python
quality = compute_quality_metrics(real_X, synthetic_X, real_y, feature_names)

quality.composite_score  # float, 0-1
quality.avg_ks_statistic  # float, lower is better
quality.corr_diff_norm  # float, lower is better
quality.ml_utility_ratio  # float, higher is better
quality.feature_reports  # List[FeatureReport]
```

### `datadiffusion.evaluation.reports`

```python
print_quality_report(quality, iteration=0, threshold=0.80)
save_quality_report(quality, save_dir="output/", iteration=0)
```

---

## Pipeline

### `datadiffusion.pipeline.SelfImprovementLoop`

```python
loop = SelfImprovementLoop(config)
result = loop.run()

result.quality.composite_score  # Best iteration's score
result.best_iteration  # Index of best iteration
result.iterations  # List[IterationResult]
```

Orchestrates the full train → generate → evaluate → adjust cycle. Each iteration produces artifacts (plots, reports, checkpoints) and logs to MLflow.

---

## Visualization

```python
from datadiffusion.visualization import (
    plot_loss_curve,
    plot_marginal_distributions,
    plot_correlation_comparison,
)

plot_loss_curve(train_losses, val_losses, save_path="loss.png")
plot_marginal_distributions(real_X, synthetic_X, feature_names, save_path="marginals.png")
plot_correlation_comparison(real_X, synthetic_X, save_path="correlations.png")
```

---

## Tracking

### `datadiffusion.tracking.mlflow_utils`

```python
outdir = setup_experiment("my_experiment")  # Sets MLflow experiment, returns output dir

with nested_run("iter_0"):
    mlflow.log_metrics({"composite": 0.92})
    mlflow.log_artifact("plot.png")
```

### `datadiffusion.tracking.logger`

```python
setup_logging(verbose=True)
logger = get_logger("datadiffusion.training")
logger.info("Training started")
```

---

## Programmatic Usage

```python
from datadiffusion.config import ExperimentConfig, TrainingConfig, GenerationConfig
from datadiffusion.data import get_dataset
from datadiffusion.models.factory import create_model_and_scheduler
from datadiffusion.training import Trainer
from datadiffusion.generation import generate_samples_ddim
from datadiffusion.evaluation import compute_quality_metrics

# Configure
config = ExperimentConfig(
    training=TrainingConfig(num_epochs=100),
    generation=GenerationConfig(sampler_type="ddim"),
)

# Load data
dataset, transformer = get_dataset("california")
real_X = dataset.tensors[0].numpy()
real_y = dataset.tensors[1].numpy()

# Train
model, scheduler = create_model_and_scheduler(config, input_size=8)
trainer = Trainer(config)
train_losses, val_losses = trainer.train(model, scheduler, dataset)

# Generate
synthetic = generate_samples_ddim(model, scheduler, num_samples=len(real_X), input_size=8)

# Evaluate in original space
real_orig = transformer.inverse_transform(real_X)
synth_orig = transformer.inverse_transform(synthetic)
quality = compute_quality_metrics(real_orig, synth_orig, real_y)
print(f"Composite: {quality.composite_score:.4f}")
```
