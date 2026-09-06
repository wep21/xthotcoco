"""OKS sigmas for the keypoint layouts xtcocotools evaluates."""

# COCO 17-point body (pycocotools defaults)
BODY = [.026, .025, .025, .035, .035, .079, .079, .072, .072,
        .062, .062, .107, .107, .087, .087, .089, .089]

# COCO-WholeBody foot (6 points), from xtcocoapi
FOOT = [.068, .066, .066, .092, .094, .094]

# COCO-WholeBody face (68) and hand (21) sigmas, from xtcocoapi
FACE = [
    .042, .043, .044, .043, .040, .035, .031, .025, .020, .023, .029, .032,
    .037, .038, .043, .041, .045, .013, .012, .011, .011, .012, .012, .011,
    .011, .013, .015, .009, .007, .007, .007, .012, .009, .008, .016, .010,
    .017, .011, .009, .011, .009, .007, .013, .008, .011, .012, .010, .034,
    .008, .008, .009, .008, .008, .007, .010, .008, .009, .009, .009, .007,
    .007, .008, .011, .008, .008, .008, .01, .008,
]
HAND = [
    .029, .022, .035, .037, .047, .026, .025, .024, .035, .018, .024, .022,
    .026, .017, .021, .021, .032, .02, .019, .022, .031,
]

WHOLEBODY = BODY + FOOT + FACE + HAND + HAND  # 133 points

# CrowdPose 14-point layout
CROWD = [.079, .079, .072, .072, .062, .062, .107, .107,
         .087, .087, .089, .089, .079, .079]

DEFAULT_SIGMAS = {
    "keypoints": BODY,
    "keypoints_wholebody": WHOLEBODY,
    "keypoints_foot": FOOT,
    "keypoints_face": FACE,
    "keypoints_lefthand": HAND,
    "keypoints_righthand": HAND,
    "keypoints_crowd": CROWD,
}
