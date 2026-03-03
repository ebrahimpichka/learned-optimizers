"""Tests for configuration loading and management."""

import os
import tempfile
import pytest
import yaml

from graph_metanet.configs.config import (
    ExperimentConfig,
    ModelConfig,
    MetaTrainingConfig,
    TaskConfig,
    WandbConfig,
    load_config,
    save_config,
    config_to_dict,
)


class TestConfigDefaults:
    def test_default_experiment_config(self):
        cfg = ExperimentConfig()
        assert cfg.seed == 42
        assert cfg.device == "cpu"
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.meta_training, MetaTrainingConfig)

    def test_default_model_config(self):
        cfg = ModelConfig()
        assert cfg.num_gnn_layers == 3
        assert cfg.use_layer_norm is True

    def test_default_wandb_config(self):
        cfg = WandbConfig()
        assert cfg.enabled is True
        assert cfg.project == "graph-metanet-optimizer"


class TestConfigIO:
    def test_save_and_load(self):
        cfg = ExperimentConfig(experiment_name="test_io", seed=123)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.experiment_name == "test_io"
            assert loaded.seed == 123

    def test_load_with_overrides(self):
        cfg = ExperimentConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            save_config(cfg, path)
            loaded = load_config(path, overrides={"seed": 999})
            assert loaded.seed == 999

    def test_load_partial_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "partial.yaml")
            with open(path, "w") as f:
                yaml.dump({"experiment_name": "partial", "model": {"num_gnn_layers": 5}}, f)
            loaded = load_config(path)
            assert loaded.experiment_name == "partial"
            assert loaded.model.num_gnn_layers == 5
            # Other defaults preserved
            assert loaded.model.use_layer_norm is True

    def test_config_to_dict(self):
        cfg = ExperimentConfig()
        d = config_to_dict(cfg)
        assert isinstance(d, dict)
        assert "model" in d
        assert "meta_training" in d
        assert d["seed"] == 42
