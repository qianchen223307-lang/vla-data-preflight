import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from vla_data_preflight.__main__ import main
from vla_data_preflight.checks import check_inputs
from vla_data_preflight.demo import make_fixture
from vla_data_preflight.report import write_report


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.tmp.name)
        self.paths = make_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self):
        return check_inputs(*self.paths, base_dir=self.root)

    def ids(self, result):
        return {d["check_id"] for d in result["diagnostics"]}

    def mutate_json(self, index, fn):
        data = json.loads(self.paths[index].read_text())
        fn(data)
        self.paths[index].write_text(json.dumps(data), encoding="utf-8")

    def mutate_config(self, fn):
        data = yaml.safe_load(self.paths[0].read_text())
        fn(data)
        self.paths[0].write_text(yaml.safe_dump(data), encoding="utf-8")

    def rewrite_samples(self, fn):
        with np.load(self.paths[3], allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        fn(arrays)
        np.savez_compressed(self.paths[3], **arrays)

    def test_valid_real_arrays(self):
        r = self.run_check()
        self.assertTrue(r["summary"]["passed"])
        self.assertEqual(r["summary"]["sample_frames"], 24)
        self.assertEqual(r["summary"]["image_streams"], 2)

    def test_injected_damage_is_distinguished(self):
        make_fixture(self.root, broken=True)
        r = self.run_check()
        self.assertEqual(r["summary"]["errors"], 6)
        self.assertTrue({"CFG_WARMUP", "MOD_OVERLAP", "MOD_CAMERA", "SAMPLE_FINITE", "SAMPLE_LENGTH", "SAMPLE_LANGUAGE"} <= self.ids(r))

    def test_safe_yaml_does_not_execute_tags(self):
        self.paths[0].write_text("!!python/object/apply:os.system ['echo forbidden']", encoding="utf-8")
        self.assertIn("INPUT_READ", self.ids(self.run_check()))

    def test_top_level_json_array_is_reported(self):
        self.paths[1].write_text("[]", encoding="utf-8")
        self.assertIn("INPUT_READ", self.ids(self.run_check()))

    def test_recursive_yaml_alias_is_reported(self):
        self.paths[0].write_text("value: &loop [*loop]", encoding="utf-8")
        self.assertIn("CFG_STRUCTURE", self.ids(self.run_check()))

    def test_empty_byte_language_is_not_accepted(self):
        self.rewrite_samples(lambda a: a.update(lang=np.array([b""] * 24)))
        self.assertIn("SAMPLE_LANGUAGE", self.ids(self.run_check()))

    def test_unused_state_dim_is_not_false_positive(self):
        self.mutate_config(lambda c: c["framework"]["action_model"].update(state_dim=14))
        self.assertTrue(self.run_check()["summary"]["passed"])
        self.mutate_config(lambda c: c["datasets"]["vla_data"].update(include_state=True))
        self.assertIn("CFG_DIM", self.ids(self.run_check()))

    def test_modality_gap_is_detected(self):
        self.mutate_json(2, lambda m: m["action"].pop("y"))
        self.assertIn("MOD_COVERAGE", self.ids(self.run_check()))

    def test_negative_slice_is_detected(self):
        self.mutate_json(2, lambda m: m["action"]["x"].update(start=-1))
        self.assertIn("MOD_SLICE", self.ids(self.run_check()))

    def test_flat_language_mapping_is_required(self):
        self.mutate_json(2, lambda m: m.update(annotation={"human": {"task_description": {"original_key": "task_index"}}}))
        self.assertIn("MOD_LANGUAGE", self.ids(self.run_check()))

    def test_snapshot_shape_checked_not_just_metadata(self):
        self.rewrite_samples(lambda a: a.update(action=a["action"][:, :6]))
        self.assertIn("SAMPLE_SHAPE", self.ids(self.run_check()))

    def test_float_image_snapshot_rejected(self):
        self.rewrite_samples(lambda a: a.update({"observation.images.image": a["observation.images.image"].astype(np.float32)}))
        self.assertIn("SAMPLE_DTYPE", self.ids(self.run_check()))

    def test_native_channel_first_metadata_maps_to_decoded_hwc(self):
        def convert(info):
            for key, feature in info["features"].items():
                if key.startswith("observation.images."):
                    feature.update(shape=[3, 32, 32], names=["channels", "height", "width"])
        self.mutate_json(1, convert)
        self.assertTrue(self.run_check()["summary"]["passed"])

    def test_missing_image_axis_names_are_reported(self):
        self.mutate_json(1, lambda i: i["features"]["observation.images.image"].pop("names"))
        self.assertIn("META_IMAGE", self.ids(self.run_check()))

    def test_pickle_snapshot_is_not_loaded(self):
        self.rewrite_samples(lambda a: a.update(action=np.array([{"do": "not execute"}], dtype=object)))
        self.assertIn("SAMPLE_READ", self.ids(self.run_check()))

    def test_missing_dataset_path_is_found(self):
        self.mutate_config(lambda c: c["datasets"]["vla_data"].update(data_root_dir="does-not-exist"))
        self.assertIn("CFG_PATH", self.ids(self.run_check()))

    def test_snapshot_diagnostics_and_strict_json(self):
        make_fixture(self.root, broken=True)
        r = self.run_check()
        self.assertTrue(all(set(d) == {"check_id", "severity", "path", "message", "hint"} for d in r["diagnostics"]))
        r["diagnostics"][0]["message"] = "<script>alert(1)</script>"
        write_report(r, self.root / "report")
        self.assertNotIn("<script>", (self.root / "report/report.html").read_text())
        json.loads((self.root / "report/report.json").read_text())

    def test_cli_checks_inputs_and_returns_failure(self):
        arguments = ["check", "--config", str(self.paths[0]), "--info", str(self.paths[1]), "--modality", str(self.paths[2]), "--samples", str(self.paths[3]), "--base-dir", str(self.root), "--output", str(self.root / "result")]
        self.assertEqual(main(arguments), 0)
        make_fixture(self.root, broken=True)
        self.assertEqual(main(arguments), 1)
        self.assertTrue((self.root / "result/report.json").is_file())


if __name__ == "__main__":
    unittest.main()
