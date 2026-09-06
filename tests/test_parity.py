"""Parity tests against xtcocotools (skipped if it is not installed)."""

import numpy as np
import pytest

from xthotcoco import COCO, COCOeval, sigmas

xt_coco = pytest.importorskip("xtcocotools.coco")
xt_eval = pytest.importorskip("xtcocotools.cocoeval")


def _wholebody_dataset(rng, n_img=20):
    sizes = {"keypoints": 17, "foot_kpts": 6, "face_kpts": 68,
             "lefthand_kpts": 21, "righthand_kpts": 21}
    images, anns, dts = [], [], []
    aid = 1
    for img_id in range(1, n_img + 1):
        images.append({"id": img_id, "width": 640, "height": 480,
                       "crowdIndex": float(rng.uniform())})
        for _ in range(int(rng.integers(1, 4))):
            x, y, w, h = rng.uniform(0, 300), rng.uniform(0, 200), 150.0, 250.0
            parts = {}
            for k, n in sizes.items():
                if k != "keypoints" and rng.random() < 0.2:
                    parts[k] = [0.0] * (n * 3)
                    continue
                xs = rng.uniform(x, x + w, n); ys = rng.uniform(y, y + h, n)
                v = rng.choice([0, 1, 2], n, p=[0.1, 0.2, 0.7]).astype(float)
                parts[k] = np.stack([np.where(v > 0, xs, 0), np.where(v > 0, ys, 0), v],
                                    1).reshape(-1).tolist()
            anns.append({"id": aid, "image_id": img_id, "category_id": 1,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                         "num_keypoints": int((np.asarray(parts["keypoints"][2::3]) == 2).sum()),
                         **parts})
            aid += 1
            det = {}
            for k, n in sizes.items():
                a = np.asarray(parts[k]).reshape(-1, 3).copy()
                a[:, :2] += rng.normal(0, 6, (n, 2)); a[:, 2] = 1
                det[k] = a.reshape(-1).tolist()
            s = float(rng.uniform(0.2, 1))
            dts.append({"image_id": img_id, "category_id": 1, **det, "score": s,
                        "wholebody_score": s * 0.9, "face_score": s * 0.8,
                        "foot_score": s, "lefthand_score": s, "righthand_score": s})
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": 1, "name": "person",
                          "keypoints": [f"k{i}" for i in range(17)], "skeleton": []}]}
    return gt, dts


def _xt_stats(gt, dts, iou_type, sig, use_area=True, test_index=None,
              gt_path="<ann_data>"):
    import contextlib, io, copy
    with contextlib.redirect_stdout(io.StringIO()):
        # xtcocotools needs a non-None annotation_file even with ann_data;
        # for keypoints_crowd it must be a real file, since get_type_result
        # re-reads the GT from that path for the crowdIndex split.
        cg = xt_coco.COCO(gt_path, test_index=test_index,
                          ann_data=copy.deepcopy(gt))
        cd = cg.loadRes(copy.deepcopy(dts))
        ev = xt_eval.COCOeval(cg, cd, iou_type, np.asarray(sig), use_area=use_area)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return np.asarray(ev.stats)


@pytest.mark.parametrize("iou_type", [
    "keypoints", "keypoints_wholebody", "keypoints_foot", "keypoints_face",
    "keypoints_lefthand", "keypoints_righthand",
])
def test_wholebody_parts_match_xtcocotools(iou_type):
    gt, dts = _wholebody_dataset(np.random.default_rng(0))
    sig = sigmas.DEFAULT_SIGMAS[iou_type]
    expected = _xt_stats(gt, dts, iou_type, sig)
    ev = COCOeval(COCO(ann_data=gt), dts, iou_type, sig)
    ev.run()
    np.testing.assert_allclose(ev.stats, expected, atol=1e-9)


def test_crowd_matches_xtcocotools(tmp_path):
    import json
    gt, dts = _wholebody_dataset(np.random.default_rng(1))
    # CrowdPose-style: 14 kpts, no area
    for a in gt["annotations"]:
        a["keypoints"] = a["keypoints"][:14 * 3]; a.pop("area")
    for d in dts:
        d["keypoints"] = d["keypoints"][:14 * 3]
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt))
    expected = _xt_stats(gt, dts, "keypoints_crowd", sigmas.CROWD,
                         use_area=False, gt_path=str(gt_path))
    ev = COCOeval(COCO(ann_data=gt), dts, "keypoints_crowd", sigmas.CROWD, use_area=False)
    ev.run()
    assert len(ev.stats) == 9
    np.testing.assert_allclose(ev.stats, expected, atol=1e-9)


def test_test_index_matches_xtcocotools():
    gt, dts = _wholebody_dataset(np.random.default_rng(2))
    idx = list(range(12))
    for d in dts:  # detections come pre-sliced, as xtcocotools expects
        d["keypoints"] = np.asarray(d["keypoints"]).reshape(-1, 3)[idx].reshape(-1).tolist()
    sig = np.asarray(sigmas.BODY)[idx]
    expected = _xt_stats(gt, dts, "keypoints", sig, test_index=idx)
    ev = COCOeval(COCO(ann_data=gt, test_index=idx), dts, "keypoints", sig)
    ev.run()
    np.testing.assert_allclose(ev.stats, expected, atol=1e-9)
