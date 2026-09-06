"""xtcocotools-compatible keypoint evaluation on top of hotcoco.

xtcocotools stores each COCO-WholeBody part under its own annotation key and
selects at eval time which parts to concatenate, which per-part OKS sigmas to
use, and which detection score key to rank by. hotcoco evaluates standard COCO
keypoints with configurable ``kpt_oks_sigmas``, so this module rewrites a
WholeBody/CrowdPose-style dataset into an equivalent standard-keypoints
dataset and delegates to hotcoco's Rust engine.

Reproduced xtcocotools behaviors:

- part iouTypes ``keypoints_wholebody`` / ``keypoints_foot`` /
  ``keypoints_face`` / ``keypoints_lefthand`` / ``keypoints_righthand``:
  part concatenation, part score keys (``wholebody_score`` etc., falling
  back to ``score``), GT ignored when the part has no visible keypoint,
  all-zero-part detections dropped
- detection area taken from the extent of the *body* keypoints regardless of
  the part being evaluated (xtcocotools' ``loadRes``)
- ``use_area=False``: 0.53*w*h replaces the GT area
- ``keypoints_crowd`` (CrowdPose): GT ignore from the ``num_keypoints`` field,
  9-stat summary with AP easy/medium/hard by image ``crowdIndex``
- ``COCO(test_index=...)``: GT keypoint-subset selection
"""

from __future__ import annotations

import contextlib
import copy
import io
import json

import numpy as np

import hotcoco

# part -> (annotation keys to concatenate, detection score key)
PART_SPECS = {
    "keypoints": (["keypoints"], "score"),
    "keypoints_wholebody": (
        ["keypoints", "foot_kpts", "face_kpts", "lefthand_kpts", "righthand_kpts"],
        "wholebody_score",
    ),
    "keypoints_foot": (["foot_kpts"], "foot_score"),
    "keypoints_face": (["face_kpts"], "face_score"),
    "keypoints_lefthand": (["lefthand_kpts"], "lefthand_score"),
    "keypoints_righthand": (["righthand_kpts"], "righthand_score"),
    "keypoints_crowd": (["keypoints"], "score"),
}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _load(obj):
    if isinstance(obj, str):
        with open(obj) as f:
            return json.load(f)
    return obj


def _concat_parts(ann, keys):
    if len(keys) == 1:
        return list(ann[keys[0]])
    out = []
    for k in keys:
        out.extend(ann[k])
    return out


class COCO:
    """Lightweight stand-in for ``xtcocotools.coco.COCO``.

    Holds the raw dataset (file path or dict) plus ``test_index``; the actual
    indexing happens in hotcoco after :func:`transform_gt`. ``loadRes`` returns
    the raw result list, to be paired with this object in :class:`COCOeval`.
    """

    def __init__(self, annotation_file=None, test_index=None, ann_data=None):
        if annotation_file is not None:
            self.dataset = _load(annotation_file)
        elif ann_data is not None:
            self.dataset = ann_data
        else:
            self.dataset = {}
        self.test_index = test_index

    def loadRes(self, resFile, resData=None):
        return Results(_load(resFile) if resData is None else resData)

    load_res = loadRes


class Results(list):
    """Detection results as returned by :meth:`COCO.loadRes`."""


def transform_gt(gt_dataset, iou_type, use_area=True, test_index=None):
    """Rewrite a WholeBody/CrowdPose-style GT dict into standard COCO form."""
    keys, _ = PART_SPECS[iou_type]
    gt = {k: v for k, v in gt_dataset.items() if k not in ("annotations", "categories")}

    anns = []
    n_kpts = None
    for a in gt_dataset["annotations"]:
        kpts = _concat_parts(a, keys)
        if test_index is not None:
            # xtcocotools.COCO(test_index=...) keeps only these keypoint rows
            # of the GT (detections are expected to come pre-sliced).
            kpts = np.asarray(kpts).reshape(-1, 3)[test_index].reshape(-1).tolist()
        n_kpts = len(kpts) // 3
        v = np.asarray(kpts[2::3])
        if iou_type == "keypoints_crowd":
            # CrowdPose ignores a GT based on the num_keypoints field, which
            # counts only fully visible (v=2) joints.
            nk = int(a["num_keypoints"])
        else:
            # xtcocotools ignores a GT when its selected part has no visible
            # keypoints; expressing that via num_keypoints matches both the
            # pycocotools and hotcoco conventions.
            nk = int(np.count_nonzero(v > 0))
        new = {
            "id": a["id"], "image_id": a["image_id"],
            "category_id": a["category_id"], "bbox": list(a["bbox"]),
            "iscrowd": a.get("iscrowd", 0),
            "keypoints": kpts,
            "num_keypoints": nk,
        }
        if use_area and "area" in a:
            new["area"] = a["area"]
        else:
            new["area"] = a["bbox"][2] * a["bbox"][3] * 0.53
        anns.append(new)
    gt["annotations"] = anns

    cats = []
    for c in gt_dataset["categories"]:
        c = copy.deepcopy(c)
        c["keypoints"] = [f"kpt_{i}" for i in range(n_kpts or 0)]
        c["skeleton"] = []
        cats.append(c)
    gt["categories"] = cats
    return gt


def transform_dt(dt_list, iou_type):
    """Rewrite WholeBody-style detections into standard-keypoints results."""
    keys, score_key = PART_SPECS[iou_type]
    out = []
    for d in dt_list:
        kpts = _concat_parts(d, keys)
        v = np.asarray(kpts[2::3])
        if iou_type != "keypoints" and not np.count_nonzero(v > 0):
            continue  # xtcocotools drops all-zero part detections
        # xtcocotools' loadRes derives the detection's area (used for the
        # medium/large area-range buckets) from the extent of the *body*
        # keypoints regardless of the part being evaluated; pass that bbox
        # explicitly so hotcoco doesn't recompute it from the part keypoints.
        body = np.asarray(d["keypoints"], dtype=float)
        x, y = body[0::3], body[1::3]
        x0, y0 = x.min(), y.min()
        out.append({
            "image_id": d["image_id"], "category_id": d["category_id"],
            "keypoints": kpts,
            "bbox": [x0, y0, x.max() - x0, y.max() - y0],
            "score": float(d.get(score_key, d["score"])),
        })
    return out


class COCOeval:
    """``xtcocotools.cocoeval.COCOeval``-compatible evaluator over hotcoco.

    ``cocoGt`` may be a :class:`COCO`, a GT file path, or a dataset dict;
    ``cocoDt`` a :class:`Results`, a results file path, or a list of result
    dicts. ``sigmas``, ``use_area`` and ``test_index`` follow xtcocotools
    (``test_index`` may also come from the :class:`COCO` object).
    """

    def __init__(self, cocoGt, cocoDt, iouType="keypoints", sigmas=None,
                 use_area=True, test_index=None):
        if iouType not in PART_SPECS:
            raise ValueError(f"unsupported iouType: {iouType}")
        self._iou_type = iouType
        self._sigmas = None if sigmas is None else [float(s) for s in sigmas]

        if isinstance(cocoGt, COCO):
            gt_raw = cocoGt.dataset
            if test_index is None:
                test_index = cocoGt.test_index
        else:
            gt_raw = _load(cocoGt)
        dt_raw = list(cocoDt) if isinstance(cocoDt, list) else _load(cocoDt)

        # image id -> crowdIndex, needed for the CrowdPose easy/medium/hard split
        self._img_crowd_index = {img["id"]: img.get("crowdIndex")
                                 for img in gt_raw.get("images", [])}
        gt_std = transform_gt(gt_raw, iouType, use_area, test_index)
        dt_std = transform_dt(dt_raw, iouType)
        self.cocoGt = hotcoco.COCO(gt_std)
        self.cocoDt = self.cocoGt.loadRes(dt_std)
        self._ev = self._make_ev()
        self._stats = None

    def _make_ev(self, img_ids=None):
        ev = hotcoco.COCOeval(self.cocoGt, self.cocoDt, "keypoints")
        if self._sigmas is not None:
            ev.params.kpt_oks_sigmas = self._sigmas
        if img_ids is not None:
            ev.params.imgIds = list(img_ids)
        return ev

    @property
    def params(self):
        return self._ev.params

    @property
    def eval(self):
        return self._ev.eval

    @property
    def stats(self):
        if self._stats is not None:
            return self._stats
        return self._ev.stats

    def evaluate(self):
        self._ev.evaluate()

    def accumulate(self):
        self._ev.accumulate()

    def split(self, first=0.01, second=0.85):
        """Split image ids into easy/medium/hard by image crowdIndex."""
        easy, mid, hard = [], [], []
        for img_id, ci in self._img_crowd_index.items():
            if ci < first:
                easy.append(img_id)
            elif ci < second:
                mid.append(img_id)
            else:
                hard.append(img_id)
        return easy, mid, hard

    def get_type_result(self, first=0.01, second=0.85):
        """CrowdPose AP on the easy/medium/hard image subsets.

        Reproduces xtcocotools.get_type_result: re-evaluate on each subset and
        take the plain mean of eval['precision'][:, :, :, 0, :] (area 'all'),
        rounded to 4 decimals.
        """
        res = []
        for subset in self.split(first, second):
            ev = self._make_ev(img_ids=subset)
            with _quiet():
                ev.evaluate()
                ev.accumulate()
            precision = np.asarray(ev.eval["precision"])
            res.append(round(float(np.mean(precision[:, :, :, 0, :])), 4))
        return res

    def summarize(self):
        if self._iou_type != "keypoints_crowd":
            self._ev.summarize()
            return
        # xtcocotools' _summarizeKps_crowd: AP/AP50/AP75, AR/AR50/AR75 on the
        # 'all' area range, then AP(easy)/AP(medium)/AP(hard) with the
        # crowdIndex split fixed at (0.2, 0.8).
        with _quiet():
            self._ev.summarize()
        s = np.asarray(self._ev.stats, dtype=float)
        stats = np.zeros(9)
        stats[0:3] = s[0:3]   # AP, AP50, AP75 (all)
        stats[3:6] = s[5:8]   # AR, AR50, AR75 (all)
        stats[6:9] = self.get_type_result(first=0.2, second=0.8)
        self._stats = stats
        names = ["AP", "AP50", "AP75", "AR", "AR50", "AR75",
                 "AP(easy)", "AP(medium)", "AP(hard)"]
        for name, val in zip(names, stats):
            print(f" {name:<10} @[ maxDets= 20 ] = {val:0.3f}")

    def run(self):
        self.evaluate()
        self.accumulate()
        self.summarize()
        return self.stats


# alias kept from the pre-packaging prototype
COCOevalXT = COCOeval
