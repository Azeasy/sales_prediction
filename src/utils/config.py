"""
Config loading and validation.

Loads base.yaml then deep-merges any policy override YAML on top.
Returns a typed Config dataclass tree for safe access throughout the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    source: str = "local"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    samples_dir: str = "data/samples"
    cleaned_dir: str = "data/cleaned"
    use_cleaned: bool = True           # True = read from cleaned_dir instead of raw_dir
    default_store_id: str = "default_store"
    skip_if_exists: bool = True
    lookback_months: Optional[int] = 4  # None = use all data; N = only last N months


@dataclass
class ApiConfig:
    base_url: str = "https://api.forecasto.ru"
    timeout: int = 30
    retries: int = 3
    backoff_factor: float = 1.0


@dataclass
class CensoringConfig:
    strategy: str = "impute"       # none | drop | impute
    rolling_window: int = 7
    use_dow_grouping: bool = True


@dataclass
class FeaturesConfig:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 28])
    target_col: str = "demand_adjusted"


@dataclass
class ModelConfig:
    type: str = "lgbm"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactsConfig:
    dir: str = "artifacts"
    model_file: str = "lgbm_model.pkl"
    features_file: str = "feature_list.json"
    feature_importance_file: str = "feature_importance.csv"


@dataclass
class LlmConfig:
    """LLM API settings for data cleaning (clean-data command)."""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"  # env var name for API key


@dataclass
class ProductMatchingConfig:
    enabled: bool = False
    embed_threshold: float = 0.92
    use_weaviate: bool = False           # True = delegate to Weaviate (docker compose up -d)
    weaviate_url: str = "http://localhost:8080"


@dataclass
class PolicyConfig:
    mode: str = "balanced"                    # service_first | balanced | waste_first
    safety_stock_multiplier: float = 1.0
    forecast_quantile: float = 0.55
    max_cover_days: Optional[int] = None      # None = use expiration_days from product
    round_up_shipment: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    censoring: CensoringConfig = field(default_factory=CensoringConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    product_matching: ProductMatchingConfig = field(default_factory=ProductMatchingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dict_to_config(d: dict) -> Config:
    """Populate Config dataclass from a flat dict (top-level keys map to sub-configs)."""

    def _from_dict(cls, data: dict):
        import inspect
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in data:
                kwargs[name] = data[name]
        return cls(**kwargs)

    return Config(
        data=_from_dict(DataConfig, d.get("data", {})),
        api=_from_dict(ApiConfig, d.get("api", {})),
        censoring=_from_dict(CensoringConfig, d.get("censoring", {})),
        features=_from_dict(FeaturesConfig, d.get("features", {})),
        model=_from_dict(ModelConfig, d.get("model", {})),
        artifacts=_from_dict(ArtifactsConfig, d.get("artifacts", {})),
        llm=_from_dict(LlmConfig, d.get("llm", {})),
        product_matching=_from_dict(ProductMatchingConfig, d.get("product_matching", {})),
        policy=_from_dict(PolicyConfig, d.get("policy", {})),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BASE_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "base.yaml"

_POLICY_CONFIG_MAP = {
    "service_first": "policy_service_first.yaml",
    "balanced": "policy_balanced.yaml",
    "waste_first": "policy_waste_first.yaml",
}


def load_config(
    base_path: str | Path | None = None,
    policy: str | None = None,
    override_path: str | Path | None = None,
) -> Config:
    """
    Load configuration.

    Args:
        base_path: Path to base YAML. Defaults to configs/base.yaml.
        policy: Policy mode name ('service_first', 'balanced', 'waste_first').
                If provided, merges the matching policy YAML on top of base.
        override_path: Explicit path to a YAML file to merge last (highest priority).

    Returns:
        Populated Config dataclass.
    """
    base_path = Path(base_path) if base_path else _BASE_CONFIG_PATH
    raw = _load_yaml(base_path)

    # Merge policy-specific config
    if policy and policy in _POLICY_CONFIG_MAP:
        policy_file = base_path.parent / _POLICY_CONFIG_MAP[policy]
        if policy_file.exists():
            raw = _deep_merge(raw, _load_yaml(policy_file))

    # Merge explicit override
    if override_path:
        raw = _deep_merge(raw, _load_yaml(Path(override_path)))

    # Env var can inject API token (not stored in config object, used by client)
    return _dict_to_config(raw)


def get_api_token() -> str:
    """
    Retrieve the Forecasto API token from the environment.
    Raises a clear error if not set.
    """
    token = os.environ.get("FORECASTO_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "FORECASTO_TOKEN environment variable is not set. "
            "Set it in your shell or in a .env file before using data.source=api."
        )
    return token
