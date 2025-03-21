## Release note for old version

- [Release note for v0](/docs/release_note/release_note_v0.md)

## Next release

## v1.0.0

- Release major version v1.0.0

We are glad to announce the release of major version v1.0.0 as OSS "AWML".

- Update CenterPoint model from base/1.0 to base/1.2
  - Dataset: test dataset of db_jpntaxi_v1 + db_jpntaxi_v2 + db_jpntaxi_v4 + db_gsm8_v1 + db_j6_v1 + db_j6_v2 + db_j6_v3 + db_j6_v5 (total frames: 3083):
  - Class mAP for center distance (0.5m, 1.0m, 2.0m, 4.0m)

| eval range: 120m     | mAP  | car  | truck | bus  | bicycle | pedestrian |
| -------------------- | ---- | ---- | ----- | ---- | ------- | ---------- |
| CenterPoint base/1.2 | 65.7 | 77.2 | 54.7  | 77.9 | 53.7    | 64.9       |
| CenterPoint base/1.1 | 64.2 | 77.0 | 52.8  | 76.7 | 51.9    | 62.7       |
| CenterPoint base/1.0 | 62.6 | 75.2 | 47.4  | 74.7 | 52.0    | 63.9       |

### Core library

- https://github.com/tier4/AWML/pull/1 Release OSS
- Dependency
  - https://github.com/tier4/AWML/pull/2
  - https://github.com/tier4/AWML/pull/5
  - https://github.com/tier4/AWML/pull/7
  - https://github.com/tier4/AWML/pull/11
- Update config
  - https://github.com/tier4/AWML/pull/8

### Tools

- Add latency measurement tools
  - https://github.com/tier4/AWML/pull/10

### Pipelines

### Projects

- CenterPoint
  - https://github.com/tier4/AWML/pull/8 Release CenterPoint base/1.2
- MobileNetv2
  - https://github.com/tier4/AWML/pull/13 Add document
- YOLOX_opt
  - https://github.com/tier4/AWML/pull/13 Add document

### Chore

- Fix docs
  - https://github.com/tier4/AWML/pull/6
  - https://github.com/tier4/AWML/pull/12
  - https://github.com/tier4/AWML/pull/17
