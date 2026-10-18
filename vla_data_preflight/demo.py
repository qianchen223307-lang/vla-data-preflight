import json
from pathlib import Path

import numpy as np
import yaml

from .checks import REFERENCE_COMMIT, check_inputs
from .report import write_report


def make_fixture(root, broken=False):
    root = Path(root)
    meta = root / "dataset/meta"
    meta.mkdir(parents=True, exist_ok=True)
    config = {"run_id": "synthetic_preflight", "seed": 42,
              "framework": {"name": "QwenOFT", "action_model": {"action_dim": 7, "state_dim": 8}},
              "datasets": {"vla_data": {"dataset_py": "lerobot_datasets", "data_root_dir": "dataset", "data_mix": "libero_all", "include_state": False, "per_device_batch_size": 2}},
              "trainer": {"max_train_steps": 100, "num_warmup_steps": 10, "gradient_accumulation_steps": 1, "learning_rate": {"base": 0.000025}}}
    camera_keys = ["observation.images.image", "observation.images.wrist_image"]
    features = {"action": {"dtype": "float32", "shape": [7]}, "observation.state": {"dtype": "float32", "shape": [8]}, **{key: {"dtype": "video", "shape": [32, 32, 3], "names": ["height", "width", "channel"], "video_info": {"video.fps": 20}} for key in camera_keys}}
    info = {"codebase_version": "v2.1", "robot_type": "franka", "fps": 20, "total_episodes": 1, "total_frames": 24, "features": features, "data_kind": "synthetic_example"}
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    state_names = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
    modality = {"action": {name: {"start": i, "end": i + 1} for i, name in enumerate(action_names)}, "state": {name: {"start": i, "end": i + 1} for i, name in enumerate(state_names)}, "video": {"primary_image": {"original_key": camera_keys[0]}, "wrist_image": {"original_key": camera_keys[1]}}, "annotation": {"human.action.task_description": {"original_key": "task_index"}}}
    t = np.arange(24, dtype=np.float32)
    arrays = {"action": np.stack([np.sin(t / 7 + k) * .1 for k in range(7)], axis=1), "observation.state": np.stack([np.cos(t / 9 + k) * .1 for k in range(8)], axis=1), "lang": np.array(["move the synthetic block to the marked area"] * 24), "task_index": np.zeros(24, dtype=np.int64)}
    yy, xx = np.indices((32, 32))
    for j, key in enumerate(camera_keys):
        images = np.zeros((24, 32, 32, 3), dtype=np.uint8)
        for i in range(24):
            images[i, ..., 0] = (xx * 4 + i * 7 + j * 20) % 256
            images[i, ..., 1] = (yy * 5 + j * 30) % 256
            images[i, ..., 2] = 100
        arrays[key] = images
    if broken:
        config["trainer"]["num_warmup_steps"] = 100
        modality["action"]["y"] = {"start": 0, "end": 2}
        modality["video"]["wrist_image"]["original_key"] = "observation.images.missing"
        arrays["action"][5, 2] = np.inf
        arrays[camera_keys[0]] = arrays[camera_keys[0]][:-1]
        arrays["lang"][3] = ""
    (root / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (meta / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(root / "samples.npz", **arrays)
    return root / "train.yaml", meta / "info.json", meta / "modality.json", root / "samples.npz"


def run_demo(root):
    root = Path(root)
    reports = {}
    for name, broken in (("valid", False), ("invalid", True)):
        fixture = root / "fixtures" / name
        inputs = make_fixture(fixture, broken)
        report = check_inputs(*inputs, base_dir=fixture)
        report["data_kind"] = "synthetic_demo"
        write_report(report, root / "results" / name)
        reports[name] = report
    (root / "results/summary.json").write_text(json.dumps({k: r["summary"] for k, r in reports.items()}, indent=2) + "\n", encoding="utf-8")

    def scenario(name, label):
        r = reports[name]
        return {"id": name, "label": label, "summary": "全部为独立生成的测试输入，未加载模型。", "metrics": [{"label": "错误数", "value": r["summary"]["errors"]}, {"label": "样本帧数", "value": r["summary"]["sample_frames"]}], "charts": [], "tables": [{"title": "检查诊断", "columns": ["检查", "路径", "说明", "修复提示"], "rows": [[d["check_id"], d["path"], d["message"], d["hint"]] for d in r["diagnostics"]]}]}
    trace = reports["valid"]["action_trace"]
    data = {"slug": "vla-data-preflight", "title": "VLA 数据与配置预检", "tagline": "用真实数组发现接口错配", "category": "研究工程工具", "data_kind": "synthetic_demo", "summary": "独立实现的 LIBERO/QwenOFT 预检：核对训练 YAML、LeRobot 元数据、modality 切片与 NPZ 样本，不启动大模型训练。",
            "metrics": [{"label": "正常样例错误", "value": reports["valid"]["summary"]["errors"]}, {"label": "异常样例错误", "value": reports["invalid"]["summary"]["errors"]}, {"label": "图像流", "value": 2}, {"label": "合成样本", "value": 24, "unit": "帧"}],
            "charts": [{"title": "实际读取的合成动作数组", "kind": "line", "labels": list(range(len(trace))), "series": [{"name": f"action[{j}]", "values": [r[j] for r in trace]} for j in range(2)], "y_label": "合成动作值"}],
            "tables": [{"title": "预检输入", "columns": ["输入", "核验内容"], "rows": [["train.yaml", "框架、维数、路径、batch、warmup、学习率"], ["meta/info.json", "feature形状、图像映射、fps"], ["meta/modality.json", "切片范围/重叠/覆盖、相机原键、语言原键"], ["samples.npz", "真实数组的T/shape/dtype/finite/语言"]]}],
            "scenarios": [scenario("valid", "正常配置"), scenario("invalid", "六类故意错配")],
            "method": [f"接口依据固定在 StarVLA 提交 {REFERENCE_COMMIT[:8]}，检查逻辑为原创实现。", "安全读取 YAML/JSON/NPZ，不执行数据注册脚本，不允许 pickle 数组。", "输出结构化诊断和可审阅的 JSON/HTML 报告；退出码支持自动化检查。"], "limitations": reports["valid"]["limitations"]}
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return reports
