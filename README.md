# xthotcoco

xtcocotools-compatible keypoint evaluation — COCO-WholeBody part metrics and
CrowdPose — running on [hotcoco](https://pypi.org/project/hotcoco/)'s Rust
engine. Pure Python; the only compiled code is hotcoco's prebuilt wheel.

```bash
pip install xthotcoco
```

## Usage

```python
from xthotcoco import COCO, COCOeval, sigmas

gt = COCO("coco_wholebody_val_v1.0.json")
dt = gt.loadRes("results.json")           # dicts with keypoints, foot_kpts, ..., wholebody_score
ev = COCOeval(gt, dt, "keypoints_wholebody", sigmas.WHOLEBODY)
ev.evaluate(); ev.accumulate(); ev.summarize()
print(ev.stats)                           # 10 COCO keypoint metrics

# CrowdPose: 9 stats incl. AP easy/medium/hard
ev = COCOeval(COCO("crowdpose_test.json"), dt, "keypoints_crowd",
              sigmas.CROWD, use_area=False)
ev.run()
```

Supported `iouType`s: `keypoints`, `keypoints_wholebody`, `keypoints_foot`,
`keypoints_face`, `keypoints_lefthand`, `keypoints_righthand`,
`keypoints_crowd`. `COCO(..., test_index=[...])` evaluates a keypoint subset.
GT/DT may be paths, dicts/lists, or the `COCO`/`Results` wrappers.

## Parity and speed

Matches xtcocotools 1.14.3 to double precision on every metric. Real
predictions — rtmlib RTMPose-WholeBody on 500 COCO val2017 images (2830
instances), COCO-WholeBody v1.0 val annotations, official sigmas:

| iouType | AP | AP50 | AP75 | xtcocotools | xthotcoco | Speedup | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| keypoints (body) | 0.721 | 0.868 | 0.789 | 0.468 s | 0.005 s | 90x | 7e-16 |
| keypoints_wholebody | 0.649 | 0.870 | 0.737 | 0.774 s | 0.008 s | 101x | 6e-16 |
| keypoints_foot | 0.734 | 0.860 | 0.776 | 0.511 s | 0.003 s | 153x | 2e-15 |
| keypoints_face | 0.822 | 0.929 | 0.882 | 0.638 s | 0.008 s | 76x | 3e-15 |
| keypoints_lefthand | 0.614 | 0.878 | 0.680 | 0.585 s | 0.004 s | 148x | 1e-15 |
| keypoints_righthand | 0.601 | 0.872 | 0.655 | 0.561 s | 0.004 s | 139x | 1e-15 |

Synthetic WholeBody / CrowdPose benchmarks at COCO-val scale (5000 images,
20-core machine) show 50–190x for the part iouTypes and 48x for
`keypoints_crowd` (which re-evaluates three image subsets for AP
easy/medium/hard, like xtcocotools), with ~1e-15 agreement on all metrics
including the 9 CrowdPose stats and `test_index` subsets. Times are for
evaluate+accumulate+summarize; the Python-side dataset rewrite happens once at
construction. Parity tests live in `tests/` and run against xtcocotools when it
is installed (`pip install xthotcoco[test]`).

## How it works

xtcocotools keeps each WholeBody part under its own key and picks parts,
sigmas and score key at eval time. xthotcoco rewrites the dataset into
standard COCO keypoints form (part concatenation, part score, xtcocotools'
body-extent detection area, `use_area=False` normalization, CrowdPose's
`num_keypoints`-based ignore) and hands it to `hotcoco.COCOeval` with
per-part `kpt_oks_sigmas`. OKS and AP are computed entirely in hotcoco's
Rust core; nothing numeric is reimplemented here.

## License

MIT. Evaluation semantics follow xtcocotools (MIT, © 2020 Sheng Jin, Can
Wang and Zhizhong Li).
