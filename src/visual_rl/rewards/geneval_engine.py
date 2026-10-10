"""Dependency-injected port of reward-server's GenEval evaluator.

Upstream: yifan123/reward-server, commit
8245af22b75fe640950a880ad31fe702a5d9a014, reward_server/gen_eval.py.
Copyright (c) 2023 Jie Liu; MIT license in licenses/reward-server-MIT.txt.

Scientific scoring, including negative dense rewards, strict scoring ignoring
exclusions, and the failed-position reward quirk, is intentionally unchanged.
"""

import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps


def build_evaluator(
    *,
    detector,
    clip_model,
    transform,
    tokenizer,
    classnames,
    device,
    inference_detector,
    zsc,
    color_workers=4,
):
    """Build a scorer without loading models or changing backend global state.

    The inference callback accepts (detector, list_of_numpy_images) and returns
    one mmdet-style per-class detection result per image, optionally with masks.
    The caller owns native inference chunking and model/device configuration.

    The returned callable accepts paired nonempty image/metadata sequences and
    returns (scores, ordinary, strict, grouped_ordinary, grouped_strict).
    Color crops use batches of 16. Use color_workers=0 for in-process loading;
    positive workers retain upstream multiprocessing behavior and requirements.
    """
    if (
        isinstance(color_workers, bool)
        or not isinstance(color_workers, int)
        or color_workers < 0
    ):
        raise ValueError("color_workers must be an integer >= 0")

    COLORS = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
    ]
    COLOR_CLASSIFIERS = {}

    # Evaluation parts

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects):
            self._image = image.convert("RGB")
            bgcolor = "#999"
            if bgcolor == "original":
                self._blank = self._image.copy()
            else:
                self._blank = Image.new("RGB", image.size, color=bgcolor)
            self._objects = objects

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            if mask is not None:
                assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                    index,
                    self._image.size[::-1],
                    mask.shape,
                )
                image = Image.composite(self._image, self._blank, Image.fromarray(mask))
            else:
                image = self._image
            image = image.crop(box[:4])
            return (transform(image), 0)

    def color_classification(image, bboxes, classname):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                clip_model,
                tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    "a photo of a {c} object",
                ],
                device,
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes), batch_size=16, num_workers=color_workers
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, clf, dataloader, device)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(box_a, box_b):
        def area_fn(box):
            return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

        i_area = area_fn(
            [
                max(box_a[0], box_b[0]),
                max(box_a[1], box_b[1]),
                min(box_a[2], box_b[2]),
                min(box_a[3], box_b[3]),
            ]
        )
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions"""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(
            np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0
        ) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def evaluate(image, objects, metadata):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    # Color check
                    colors = color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    # Relative position check
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {len(objects[classname])}"
                )
        return correct, "\n".join(reason)

    def evaluate_reward(image, objects, metadata):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}=={req['count']}, found {len(found_objects)}"
                )
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    # Color check
                    colors = color_classification(image, found_objects, classname)
                    rewards.append(
                        1
                        - abs(req["count"] - colors.count(req["color"])) / req["count"]
                    )
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    # Relative position check
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        # Upstream appends this even after a failed position check.
                        rewards.append(1.0)
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)

    def evaluate_image(image_pils, metadatas, only_strict):
        results = inference_detector(
            detector, [np.array(image_pil) for image_pil in image_pils]
        )
        ret = []
        for result, image_pil, metadata in zip(results, image_pils, metadatas):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            # Determine bounding boxes to keep
            confidence_threshold = (
                THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
            )
            for index, classname in enumerate(classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[
                    bbox[index][ordering, 4] > confidence_threshold
                ]  # Threshold
                ordering = ordering[
                    :MAX_OBJECTS
                ].tolist()  # Limit number of detected objects per class
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append(
                        (
                            bbox[index][max_obj],
                            None if segm is None else segm[index][max_obj],
                        )
                    )
                    ordering = [
                        obj
                        for obj in ordering
                        if NMS_THRESHOLD == 1
                        or compute_iou(bbox[index][max_obj], bbox[index][obj])
                        < NMS_THRESHOLD
                    ]
                if not detected[classname]:
                    del detected[classname]
            # Evaluate
            is_strict_correct, score, reason = evaluate_reward(
                image, detected, metadata
            )
            if only_strict:
                is_correct = False
            else:
                is_correct, _ = evaluate(image, detected, metadata)
            ret.append(
                {
                    "tag": metadata["tag"],
                    "prompt": metadata["prompt"],
                    "correct": is_correct,
                    "strict_correct": is_strict_correct,
                    "score": score,
                    "reason": reason,
                    "metadata": json.dumps(metadata),
                    "details": json.dumps(
                        {
                            key: [box.tolist() for box, _ in value]
                            for key, value in detected.items()
                        }
                    ),
                }
            )
        return ret

    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1

    @torch.no_grad()
    def compute_geneval(images, metadatas, only_strict=False):
        if len(images) != len(metadatas):
            raise ValueError("images and metadatas must have matching lengths")
        if len(images) == 0:
            raise ValueError("images and metadatas must be nonempty")
        required_keys = [
            "single_object",
            "two_object",
            "counting",
            "colors",
            "position",
            "color_attr",
        ]
        scores = []
        strict_rewards = []
        grouped_strict_rewards = defaultdict(list)
        rewards = []
        grouped_rewards = defaultdict(list)
        results = evaluate_image(images, metadatas, only_strict=only_strict)
        for result in results:
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict_rewards[key].append(-10.0)
                    grouped_rewards[key].append(-10.0)
                else:
                    grouped_strict_rewards[tag].append(
                        1.0 if result["strict_correct"] else 0.0
                    )
                    grouped_rewards[tag].append(1.0 if result["correct"] else 0.0)
        return (
            scores,
            rewards,
            strict_rewards,
            dict(grouped_rewards),
            dict(grouped_strict_rewards),
        )

    return compute_geneval
