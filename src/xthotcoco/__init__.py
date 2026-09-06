"""xthotcoco: xtcocotools-compatible keypoint evaluation on hotcoco.

Drop-in for the xtcocotools evaluation workflow::

    from xthotcoco import COCO, COCOeval, sigmas

    gt = COCO("wholebody_val.json")
    dt = gt.loadRes("results.json")
    ev = COCOeval(gt, dt, "keypoints_wholebody", sigmas.WHOLEBODY)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(ev.stats)
"""

from . import sigmas
from .eval import (
    COCO,
    COCOeval,
    COCOevalXT,
    PART_SPECS,
    Results,
    transform_dt,
    transform_gt,
)

__version__ = "0.1.0"

__all__ = [
    "COCO", "COCOeval", "COCOevalXT", "PART_SPECS", "Results",
    "transform_dt", "transform_gt", "sigmas", "__version__",
]
