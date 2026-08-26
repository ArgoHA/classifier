"""Both configs must expose every key the code reads."""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# config_bench.yaml is gitignored (machine-specific paths) but is the config the
# regression benchmark runs on, so it is checked whenever it is present - a key missing
# only there fails the benchmark and nothing else.
CONFIGS = [
    p
    for p in (
        REPO_ROOT / "config.yaml",
        REPO_ROOT / "config_bench.yaml",
        REPO_ROOT / "img_clf" / "config" / "default.yaml",
    )
    if p.is_file()
]

REQUIRED_TRAIN_KEYS = [
    "root",
    "pretrained",
    "data_path",
    "path_to_save",
    "label_to_name",
    "img_size",
    "amp_enabled",
    "amp_dtype",
    "decision_metrics",
    "batch_size",
    "b_accum_steps",
    "epochs",
    "label_smoothing",
    "layers_to_train",
    "num_workers",
    "base_lr",
    "max_lr_mult",
    "weight_decay",
    "betas",
    "cycler_pct_start",
    "augs",
    "seed",
]
REQUIRED_AUG_KEYS = ["mixup_alpha", "cutmix_alpha", "mixup_prob", "mixup_switch_prob"]
REQUIRED_EXPORT_KEYS = [
    "half",
    "max_batch_size",
    "opt_batch_size",
    "dynamic_input",
    "formats",
    "simplify",
    "parity",
    "parity_images",
]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_has_every_key_the_code_reads(path):
    cfg = yaml.safe_load(path.read_text())
    for key in REQUIRED_TRAIN_KEYS:
        assert key in cfg["train"], f"{path.name} missing train.{key}"
    for key in REQUIRED_AUG_KEYS:
        assert key in cfg["train"]["augs"], f"{path.name} missing train.augs.{key}"
    for key in REQUIRED_EXPORT_KEYS:
        assert key in cfg["export"], f"{path.name} missing export.{key}"
    assert "formats" in cfg["bench"]
    assert "split" in cfg["check_errors"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_amp_dtype_is_supported(path):
    cfg = yaml.safe_load(path.read_text())
    assert cfg["train"]["amp_dtype"] in ("bfloat16", "float16")


def test_packaged_template_carries_no_dev_paths():
    """The shipped default must not point at this machine."""
    cfg = yaml.safe_load((REPO_ROOT / "img_clf" / "config" / "default.yaml").read_text())
    assert "/home/" not in str(cfg["train"]["root"])


def _config_accesses():
    """Every `cfg.<section>.<key>` attribute access in the package.

    Attribute access only: `cfg.train.get("x", default)` is optional by design, so it is
    deliberately not required to exist.
    """
    pattern = re.compile(r"\bcfg\.(train|export|split|bench|check_errors)\.([a-z_][a-z0-9_]*)")
    found = {}
    for path in (REPO_ROOT / "img_clf").rglob("*.py"):
        for section, key in pattern.findall(path.read_text()):
            if key == "get":
                continue
            found.setdefault(section, set()).add(key)
    return found


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_accessed_key_exists(path):
    """Catches a key the code reads but no config defines.

    That failure mode otherwise shows up as an omegaconf ConfigAttributeError partway
    through a command - `export.dynamic_input` did exactly that.
    """
    cfg = yaml.safe_load(path.read_text())
    missing = []
    for section, keys in _config_accesses().items():
        for key in sorted(keys):
            if key not in (cfg.get(section) or {}):
                missing.append(f"{section}.{key}")
    assert not missing, f"{path.name} does not define: {missing}"


class _DupKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with the same key twice."""


def _no_duplicate_keys(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key {key!r} on line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DupKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_duplicate_keys(path):
    """PyYAML keeps the *last* of two identical keys, silently.

    Two people (or two passes) adding the same setting leaves both lines in place, and the
    one you read while debugging may not be the one in effect.
    """
    yaml.load(path.read_text(), Loader=_DupKeyLoader)
