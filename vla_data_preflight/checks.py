"""Validate concrete configuration, metadata, and real-array NPZ snapshots."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

REFERENCE_COMMIT = "0ed0aad2c83f587714f6167ef60cf7218b786590"


def _at(data, path, default=None):
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_inputs(config_path, info_path, modality_path, samples_path, base_dir=None):
    diagnostics, stats = [], {"sample_frames": 0, "image_streams": 0, "action_dim": 7, "state_dim": 8}
    trace = []

    def add(check_id, severity, path, message, hint):
        diagnostics.append(dict(check_id=check_id, severity=severity, path=str(path), message=message, hint=hint))

    def read_mapping(path, is_yaml=False):
        try:
            content = Path(path).read_text(encoding="utf-8-sig")
            data = yaml.safe_load(content) if is_yaml else json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("top-level value must be a mapping")
            return data
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            add("INPUT_READ", "error", Path(path).name, f"Cannot read mapping: {exc}", "Supply readable YAML/JSON with an object at the top level.")
            return {}

    config = read_mapping(config_path, True)
    info = read_mapping(info_path)
    modality = read_mapping(modality_path)
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    if _at(config, "framework.name") != "QwenOFT":
        add("CFG_PROFILE", "error", "framework.name", "This v1 profile supports QwenOFT with the LIBERO contract only.", "Use the declared profile; other VLA frameworks need a separate adapter.")
    if _at(config, "datasets.vla_data.dataset_py") != "lerobot_datasets":
        add("CFG_PROFILE", "error", "datasets.vla_data.dataset_py", "Expected lerobot_datasets.", "Select the supported StarVLA data loader.")
    action_dim = _at(config, "framework.action_model.action_dim")
    if not _positive_int(action_dim) or action_dim != 7:
        add("CFG_DIM", "error", "framework.action_model.action_dim", "LIBERO/OFT action_dim must be 7.", "Match the seven LIBERO action coordinates; this tool does not convert action spaces.")
    include_state = _at(config, "datasets.vla_data.include_state", False)
    if not isinstance(include_state, bool):
        add("CFG_STATE", "error", "datasets.vla_data.include_state", "include_state must be boolean.", "Use true or false, not a string.")
    if include_state is True and _at(config, "framework.action_model.state_dim") != 8:
        add("CFG_DIM", "error", "framework.action_model.state_dim", "Selected LIBERO state input requires state_dim=8.", "Set 8 when using this state layout; an unused state_dim is not validated.")
    for key in ("datasets.vla_data.per_device_batch_size", "trainer.max_train_steps", "trainer.gradient_accumulation_steps"):
        if not _positive_int(_at(config, key)):
            add("CFG_NUMBER", "error", key, "Expected a positive integer.", "Use an integer greater than zero.")
    warmup, total = _at(config, "trainer.num_warmup_steps"), _at(config, "trainer.max_train_steps")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0 or (_positive_int(total) and warmup >= total):
        add("CFG_WARMUP", "error", "trainer.num_warmup_steps", "Warmup must be an integer in [0, max_train_steps).", "Keep a nonempty post-warmup training interval.")
    lr = _at(config, "trainer.learning_rate.base")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        add("CFG_LR", "error", "trainer.learning_rate.base", "Base learning rate must be positive and finite.", "Use a numeric YAML value, for example 0.000025.")
    data_root = _at(config, "datasets.vla_data.data_root_dir")
    if not isinstance(data_root, str) or not data_root.strip():
        add("CFG_PATH", "error", "datasets.vla_data.data_root_dir", "Dataset root must be a nonempty path.", "Point to the local dataset parent used by the training recipe.")
    else:
        root_path = Path(data_root) if Path(data_root).is_absolute() else base_dir / data_root
        if not root_path.is_dir():
            add("CFG_PATH", "error", "datasets.vla_data.data_root_dir", f"Local dataset directory does not exist: {data_root}", "Resolve paths against --base-dir (default: current working directory).")
    mix = _at(config, "datasets.vla_data.data_mix")
    if not isinstance(mix, str) or not mix.strip():
        add("CFG_MIX", "error", "datasets.vla_data.data_mix", "data_mix must be a nonempty string.", "Supply the name registered by your StarVLA data config; registry execution is outside this audit.")
    try:
        encoded_config = json.dumps(config, default=str)
    except (ValueError, TypeError):
        encoded_config = ""
        add("CFG_STRUCTURE", "error", "config", "Recursive or non-serializable YAML structure is unsupported.", "Use ordinary YAML mappings, lists and scalar values without recursive aliases.")
    if "<<TODO" in encoded_config or "path/to/" in encoded_config:
        add("CFG_PLACEHOLDER", "error", "config", "Unresolved TODO/path placeholder detected.", "Replace template placeholders before launching a training job.")
    features = info.get("features", {})
    if not isinstance(features, dict):
        features = {}
        add("META_FEATURES", "error", "info.features", "features must be an object.", "Use the native LeRobot info.json features mapping.")
    widths = {"action": 7, "observation.state": 8}
    for key, width in widths.items():
        feature = features.get(key, {})
        if not isinstance(feature, dict) or feature.get("shape") != [width]:
            add("META_DIM", "error", f"info.features.{key}", f"Expected shape [{width}] for this LIBERO profile.", "Export the declared action/state layout; state is retained in metadata even when not used by the model.")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        add("META_FPS", "error", "info.fps", "fps must be a positive finite number.", "Supply the actual dataset frame rate; no synchronization is inferred.")
    for section, original in (("action", "action"), ("state", "observation.state")):
        fields = modality.get(section, {})
        used = set()
        if not isinstance(fields, dict) or not fields:
            add("MOD_FIELDS", "error", f"modality.{section}", "Expected a nonempty mapping of named slices.", "Use the native flat modality schema.")
            continue
        for name, spec in fields.items():
            location = f"modality.{section}.{name}"
            if not isinstance(spec, dict):
                add("MOD_SLICE", "error", location, "Slice must be an object.", "Provide start, end and optional original_key.")
                continue
            start, end = spec.get("start"), spec.get("end")
            key = spec.get("original_key", original)
            valid = (isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and 0 <= start < end <= widths[original] and key == original)
            if not valid:
                add("MOD_SLICE", "error", location, f"Invalid slice or source key; expected 0 <= start < end <= {widths[original]} on {original}.", "Use half-open, in-bounds slices and the declared vector source.")
                continue
            current = set(range(start, end))
            if used & current:
                add("MOD_OVERLAP", "error", location, "Slice overlaps another selected coordinate.", "Each vector coordinate must be represented once.")
            used |= current
        if used != set(range(widths[original])):
            add("MOD_COVERAGE", "error", f"modality.{section}", "Slices do not cover the declared vector exactly.", "Cover all coordinates without gaps or overlaps.")
    annotation = modality.get("annotation", {})
    language = annotation.get("human.action.task_description", {}) if isinstance(annotation, dict) else {}
    if not isinstance(language, dict) or language.get("original_key") != "task_index":
        add("MOD_LANGUAGE", "error", "modality.annotation.human.action.task_description", "Expected flat annotation key mapped to task_index.", "Use the LIBERO flat key, not a nested human/task_description object.")
    video = modality.get("video", {})
    cameras = {}
    if not isinstance(video, dict) or not video:
        add("MOD_VIDEO", "error", "modality.video", "At least one video stream is required.", "Map video names to native info.json image features.")
        video = {}
    for name, spec in video.items():
        key = spec.get("original_key") if isinstance(spec, dict) else None
        feature = features.get(key, {}) if isinstance(key, str) else {}
        if not isinstance(feature, dict) or feature.get("dtype") not in ("video", "image"):
            add("MOD_CAMERA", "error", f"modality.video.{name}", "Camera original_key does not resolve to an image/video feature.", "Match observation.images.* keys in info.json exactly.")
            continue
        shape = feature.get("shape")
        names = feature.get("names")
        named_axes = isinstance(names, list) and len(names) == 3 and len(set(str(n) for n in names)) == 3 and "height" in names and "width" in names and ("channel" in names or "channels" in names)
        if not isinstance(shape, list) or len(shape) != 3 or not all(_positive_int(n) for n in shape) or not named_axes:
            add("META_IMAGE", "error", f"info.features.{key}", "Expected three positive shape dimensions with named height, width and channel(s) axes.", "Keep native feature names; the decoded snapshot itself must be HWC.")
            continue
        channel_name = "channel" if "channel" in names else "channels"
        canonical_shape = [shape[names.index("height")], shape[names.index("width")], shape[names.index(channel_name)]]
        if canonical_shape[-1] != 3:
            add("META_IMAGE", "error", f"info.features.{key}", "This adapter requires three decoded image channels.", "Use three-channel samples, or implement a separate channel adapter.")
            continue
        cameras[key] = canonical_shape
    stats["image_streams"] = len(cameras)
    try:
        with np.load(samples_path, allow_pickle=False) as samples:
            if "action" not in samples:
                add("SAMPLE_REQUIRED", "error", "samples.action", "Action array is missing.", "Create an actual-array NPZ snapshot using the documented adapter.")
                length = None
            else:
                shape = samples["action"].shape
                length = shape[0] if len(shape) else 0
                stats["sample_frames"] = int(length)
            required = {"action": (7,), "observation.state": (8,), **{key: tuple(shape) for key, shape in cameras.items()}}
            for key, trailing in required.items():
                if key not in samples:
                    if key != "action":
                        add("SAMPLE_REQUIRED", "error", f"samples.{key}", "Required snapshot array is missing.", "Include metadata-declared arrays in the NPZ snapshot.")
                    continue
                array = samples[key]
                if array.ndim != len(trailing) + 1 or tuple(array.shape[1:]) != trailing:
                    add("SAMPLE_SHAPE", "error", f"samples.{key}", f"Expected [T,{','.join(map(str, trailing))}], got {array.shape}.", "Export actual arrays with the metadata-declared shape.")
                if not array.shape or array.shape[0] == 0 or (length is not None and array.shape[0] != length):
                    add("SAMPLE_LENGTH", "error", f"samples.{key}", "Snapshot arrays must have the same nonzero T.", "Select the same sampled frames for all modalities.")
                if key in cameras:
                    if array.dtype != np.dtype("uint8"):
                        add("SAMPLE_DTYPE", "error", f"samples.{key}", "Image snapshots must be uint8.", "Keep real decoded pixels; do not substitute shape-only descriptors.")
                elif array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                    add("SAMPLE_FINITE", "error", f"samples.{key}", "Vector snapshot must be numeric and finite.", "Inspect NaN/Inf and fix the source data or exporter.")
                if key == "action" and array.ndim == 2 and array.dtype.kind in "fiu":
                    trace = [[float(v) if np.isfinite(v) else None for v in row] for row in array[:256]]
            for key in ("lang", "task_index"):
                if key not in samples:
                    add("SAMPLE_LANGUAGE", "error", f"samples.{key}", "Language snapshot array is missing.", "Include resolved lang strings and their integer task_index values.")
                    continue
                arr = samples[key]
                if arr.ndim != 1 or len(arr) == 0 or (length is not None and len(arr) != length):
                    add("SAMPLE_LANGUAGE", "error", f"samples.{key}", "Expected a nonempty [T] array matching action.", "Align language/task indices with the selected frames.")
                elif key == "lang" and (arr.dtype.kind != "U" or any(not str(s).strip() for s in arr)):
                    add("SAMPLE_LANGUAGE", "error", "samples.lang", "Language instructions must be nonempty strings.", "Resolve task_index using the real task table during export.")
                elif key == "task_index" and (arr.dtype.kind not in "iu" or np.any(arr < 0)):
                    add("SAMPLE_LANGUAGE", "error", "samples.task_index", "Task indices must be nonnegative integers.", "Use the dataset's real task indices.")
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        add("SAMPLE_READ", "error", Path(samples_path).name, f"Cannot read safe NPZ snapshot: {exc}", "Use np.savez/np.savez_compressed with numeric and Unicode arrays; object/pickle arrays are unsupported.")
    errors, warnings = (sum(d["severity"] == s for d in diagnostics) for s in ("error", "warning"))
    return {"schema": "vla-preflight-report-v1", "data_kind": "user_input", "reference_commit": REFERENCE_COMMIT,
            "summary": {**stats, "errors": errors, "warnings": warnings, "passed": errors == 0}, "diagnostics": diagnostics,
            "action_trace": trace, "limitations": ["Only the documented LIBERO/QwenOFT contract is supported.", "NPZ is this tool's snapshot adapter, not a native LeRobot storage format.", "No Parquet/MP4 decoding, registry execution, model loading, training, success-rate or energy measurement.", "A finite sample snapshot does not establish full-dataset quality or sensor synchronization."]}
