# Interface sources and attribution

Fixed reference: `yifanwu-victor/starVLA` commit `0ed0aad2c83f587714f6167ef60cf7218b786590`, a fork of [starVLA/starVLA](https://github.com/starVLA/starVLA).

- [README](https://github.com/yifanwu-victor/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/README.md): raw data boundaries and independent component design.
- [LIBERO data config](https://github.com/yifanwu-victor/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/examples/simBenchmarks/LIBERO/train_files/data_registry/data_config.py): seven action coordinates, eight state coordinates, named camera and language fields.
- [LIBERO modality](https://github.com/yifanwu-victor/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/examples/simBenchmarks/LIBERO/train_files/modality.json): flat annotation key, half-open vector slices and camera original keys.
- [Data loading entry](https://github.com/yifanwu-victor/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/dataloader/lerobot_datasets.py): named mixture and data-root contract.
- [Dataset metadata reader](https://github.com/yifanwu-victor/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/dataloader/gr00t_lerobot/datasets.py): native image shape axes are located through feature names; decoded frames use HWC.
- [Upstream LICENSE](https://github.com/yifanwu-victor/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/LICENSE): source remains under its own terms. No upstream files are copied or relicensed here.

Original additions: the complete Python checks/report/CLI code, deterministic mathematical-array generator, sample-NPZ audit contract, tests and documentation. The snapshots are not original LeRobot episode files and are not claimed as collected robotics data. This repository does not imply upstream affiliation, research authorship or endorsement.
