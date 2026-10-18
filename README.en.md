# VLA Data Preflight

[Live demo](https://qianchen223307-lang.github.io/vla-data-preflight/) · [中文](README.md)

An original CPU-only checker for a **limited, pinned StarVLA LIBERO/QwenOFT contract**. It checks actual training YAML, native LeRobot info/modality JSON and an actual-array NPZ snapshot. Contributions are the checker, CLI, adapter, diagnostics, fixtures, tests and demo; no StarVLA source code is copied.

```bash
python -m pip install -e .
python -m vla_data_preflight demo
python -m unittest discover -s tests -v
python -m vla_data_preflight check --config fixtures/valid/train.yaml --info fixtures/valid/dataset/meta/info.json --modality fixtures/valid/dataset/meta/modality.json --samples fixtures/valid/samples.npz --base-dir fixtures/valid --output results/check
```

Replace those paths with real inputs. Relative training dataset paths resolve against `--base-dir`, defaulting to the current working directory. Inputs are read-only. Checks cover model profile/action dimension, positive training values, warmup, local data root, metadata shapes/fps, exact vector slice coverage, camera mappings and flat LIBERO language metadata. Native image shapes are interpreted through their height/width/channel(s) axis names, supporting named CHW and HWC metadata; decoded NPZ images remain HWC.

**NPZ is this tool's explicit adapter, not native LeRobot storage.** Export actual arrays using your existing reader: `action [T,7]`, `observation.state [T,8]`, image original keys as `uint8 [T,H,W,3]`, resolved nonempty `lang [T]` strings and nonnegative integer `task_index [T]`. All arrays share positive T. Metadata and pixel shapes must agree; numeric vectors must be finite. State stays in the audit snapshot even when the model does not use it. Unused `state_dim` is not checked against the model config.

Results are strict JSON and self-contained HTML with `check_id,severity,path,message,hint`. Exit codes: 0 passed, 1 invalid inputs/data, 2 CLI usage. YAML loading is safe and NPZ pickle/object arrays are rejected. The actual deterministic demo yields 0 errors for valid inputs and 6 for injected warmup/overlap/camera/Inf/length/language defects. Curves retain at most the first 256 snapshot frames.

No direct Parquet/MP4 parsing, registry execution, weights, model, simulator or robot. A nonempty `data_mix` is not proof that its registry entry exists. No model quality, full-dataset quality, synchronization or energy claims are made. All demo inputs are synthetic. See [SOURCES.md](SOURCES.md) and [AI_USE.md](AI_USE.md).
