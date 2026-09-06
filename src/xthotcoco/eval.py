"""xtcocotools-compatible keypoint evaluation on top of hotcoco."""

from __future__ import annotations

import contextlib
import copy
import io
import json

import numpy as np

import hotcoco

# iouType -> (annotation keys to concatenate, detection score key)
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
    out = []
    for k in keys:
        out.extend(ann[k])
    return out


class COCO:
    """Stand-in for ``xtcocotools.coco.COCO``: holds the raw dataset."""

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
    pass


def transform_gt(gt_dataset, iou_type, use_area=True, test_index=None):
    keys, _ = PART_SPECS[iou_type]
    gt = {k: v for k, v in gt_dataset.items() if k not in ("annotations", "categories")}

    anns = []
    n_kpts = 0
    for a in gt_dataset["annotations"]:
        kpts = _concat_parts(a, keys)
        if test_index is not None:
            kpts = np.asarray(kpts).reshape(-1, 3)[test_index].reshape(-1).tolist()
        n_kpts = len(kpts) // 3
        v = np.asarray(kpts[2::3])
        if iou_type == "keypoints_crowd":
            nk = int(a["num_keypoints"])  # CrowdPose: v == 2 count from the file
        else:
            nk = int(np.count_nonzero(v > 0))  # xtcocotools: any visible part kpt
        anns.append({
            "id": a["id"], "image_id": a["image_id"],
            "category_id": a["category_id"], "bbox": list(a["bbox"]),
            "iscrowd": a.get("iscrowd", 0),
            "keypoints": kpts,
            "num_keypoints": nk,
            "area": a["area"] if use_area and "area" in a
            else a["bbox"][2] * a["bbox"][3] * 0.53,
        })
    gt["annotations"] = anns

    cats = []
    for c in gt_dataset["categories"]:
        c = copy.deepcopy(c)
        c["keypoints"] = [f"kpt_{i}" for i in range(n_kpts)]
        c["skeleton"] = []
        cats.append(c)
    gt["categories"] = cats
    return gt


def transform_dt(dt_list, iou_type):
    keys, score_key = PART_SPECS[iou_type]
    out = []
    for d in dt_list:
        kpts = _concat_parts(d, keys)
        if iou_type != "keypoints" and not np.count_nonzero(np.asarray(kpts[2::3]) > 0):
            continue
        # xtcocotools derives the detection area from the body keypoints'
        # extent whatever the part; pass it as bbox so hotcoco keeps it.
        body = np.asarray(d["keypoints"], dtype=float)
        x, y = body[0::3], body[1::3]
        out.append({
            "image_id": d["image_id"], "category_id": d["category_id"],
            "keypoints": kpts,
            "bbox": [x.min(), y.min(), x.max() - x.min(), y.max() - y.min()],
            "score": float(d.get(score_key, d["score"])),
        })
    return out


class COCOeval:
    """``xtcocotools.cocoeval.COCOeval``-compatible evaluator over hotcoco."""

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

        self._img_crowd_index = {img["id"]: img.get("crowdIndex")
                                 for img in gt_raw.get("images", [])}
        self.cocoGt = hotcoco.COCO(transform_gt(gt_raw, iouType, use_area, test_index))
        self.cocoDt = self.cocoGt.loadRes(transform_dt(dt_raw, iouType))
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
        return self._ev.stats if self._stats is None else self._stats

    def evaluate(self):
        self._ev.evaluate()

    def accumulate(self):
        self._ev.accumulate()

    def split(self, first=0.01, second=0.85):
        easy, mid, hard = [], [], []
        for img_id, ci in self._img_crowd_index.items():
            (easy if ci < first else mid if ci < second else hard).append(img_id)
        return easy, mid, hard

    def get_type_result(self, first=0.01, second=0.85):
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
        with _quiet():
            self._ev.summarize()
        s = np.asarray(self._ev.stats, dtype=float)
        stats = np.zeros(9)
        stats[0:3] = s[0:3]  # AP, AP50, AP75
        stats[3:6] = s[5:8]  # AR, AR50, AR75
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


COCOevalXT = COCOeval
