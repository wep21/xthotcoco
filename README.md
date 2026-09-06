# xthotcoco

xtcocotools-compatible keypoint evaluation (COCO-WholeBody, CrowdPose) on
[hotcoco](https://pypi.org/project/hotcoco/)'s Rust engine. Pure Python.

```bash
pip install xthotcoco
```

## Usage

```python
from xthotcoco import COCO, COCOeval, sigmas

gt = COCO("coco_wholebody_val_v1.0.json")
dt = gt.loadRes("results.json")
ev = COCOeval(gt, dt, "keypoints_wholebody", sigmas.WHOLEBODY)
ev.evaluate(); ev.accumulate(); ev.summarize()
print(ev.stats)

# CrowdPose
ev = COCOeval(COCO("crowdpose_test.json"), dt, "keypoints_crowd",
              sigmas.CROWD, use_area=False)
ev.run()
```

iouTypes: `keypoints`, `keypoints_wholebody`, `keypoints_foot`,
`keypoints_face`, `keypoints_lefthand`, `keypoints_righthand`,
`keypoints_crowd`. `COCO(..., test_index=[...])` evaluates a keypoint subset.

## Parity and speed

rtmlib RTMPose-WholeBody on 500 COCO val2017 images, COCO-WholeBody v1.0 val:

| iouType | AP | xtcocotools | xthotcoco | max diff |
|---|---:|---:|---:|---:|
| keypoints | 0.721 | 0.468 s | 0.005 s | 7e-16 |
| keypoints_wholebody | 0.649 | 0.774 s | 0.008 s | 6e-16 |
| keypoints_foot | 0.734 | 0.511 s | 0.003 s | 2e-15 |
| keypoints_face | 0.822 | 0.638 s | 0.008 s | 3e-15 |
| keypoints_lefthand | 0.614 | 0.585 s | 0.004 s | 1e-15 |
| keypoints_righthand | 0.601 | 0.561 s | 0.004 s | 1e-15 |

All metrics match xtcocotools to double precision; `tests/` checks this
against xtcocotools when it is installed.

## How it works

The WholeBody/CrowdPose dataset is rewritten into standard COCO keypoints
form (part concatenation, part score, area handling) and passed to
`hotcoco.COCOeval` with per-part `kpt_oks_sigmas`. OKS and AP are computed
in hotcoco.

## License

MIT. Evaluation semantics follow xtcocotools (MIT).
