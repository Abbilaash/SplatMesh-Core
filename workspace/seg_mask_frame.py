"""
Step 2: Edge Filter Mockup (ONNX Runtime / NPU-friendly)
--------------------------------------------------------
This version removes the Ultralytics/PyTorch runtime dependency and runs a
YOLOv8 segmentation ONNX model through ONNX Runtime.

Why this is the right shape for Snapdragon X:
- The model can execute through a hardware-backed ONNX Runtime execution
  provider such as QNN on Snapdragon X systems.
- The masking and morphology steps stay in OpenCV and NumPy.
- The script can fall back to CPU if the requested NPU provider is not
  installed on the machine.

Expected model:
- Export a YOLOv8 segmentation model to ONNX, then place the resulting
  .onnx file next to this script or pass its path via --model.

Usage:
  python seg_mask_frame.py --source frame.jpg --model yolov8n-seg.onnx --out masked_frame.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - clearer runtime guidance
    raise SystemExit(
        "onnxruntime is required. Install an ONNX Runtime build that includes "
        "your target provider, such as QNN on Snapdragon X."
    ) from exc


# COCO class IDs we treat as dynamic. These are the classes that most often
# corrupt a static reconstruction if they remain visible between frames.
DYNAMIC_CLASS_IDS = {0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19}


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    """Resize with aspect ratio preservation and pad to a square canvas."""
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return image, {
        "ratio": scale,
        "pad_left": left,
        "pad_top": top,
        "new_unpad": new_unpad,
        "input_shape": new_shape,
    }


def preprocess(frame, input_size):
    """Prepare a BGR frame for ONNX Runtime inference."""
    padded, meta = letterbox(frame, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, meta


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def xywh_to_xyxy(boxes):
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def box_iou(box, boxes):
    """IoU between one box and a set of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(box_area + boxes_area - inter, 1e-9)
    return inter / union


def nms(boxes, scores, iou_threshold):
    """Pure NumPy NMS for a small number of detections."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        index = order[0]
        keep.append(index)
        if order.size == 1:
            break
        ious = box_iou(boxes[index], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def parse_outputs(outputs):
    """Find prediction and prototype tensors from an exported YOLOv8-seg model."""
    prediction = None
    proto = None

    for output in outputs:
        array = np.asarray(output)
        if array.ndim == 4:
            proto = np.squeeze(array, axis=0) if array.shape[0] == 1 else array
        elif array.ndim == 3:
            prediction = np.squeeze(array, axis=0) if array.shape[0] == 1 else array

    if prediction is None or proto is None:
        raise RuntimeError("Could not identify YOLOv8 segmentation outputs from the ONNX model.")

    if prediction.shape[0] in (84, 116):
        prediction = prediction.T

    if proto.ndim != 3:
        raise RuntimeError(f"Unexpected prototype tensor shape: {proto.shape}")

    return prediction, proto


def decode_detections(prediction, proto, input_shape, conf_thres, iou_thres):
    """Convert raw YOLOv8-seg outputs into filtered boxes and mask coeffs."""
    proto_channels = proto.shape[0]
    if prediction.shape[1] < 4 + proto_channels + 1:
        raise RuntimeError(f"Unexpected prediction tensor shape: {prediction.shape}")

    class_scores = prediction[:, 4 : prediction.shape[1] - proto_channels]
    mask_coeffs = prediction[:, prediction.shape[1] - proto_channels :]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    dynamic_mask = (scores >= conf_thres) & np.isin(class_ids, list(DYNAMIC_CLASS_IDS))
    if not np.any(dynamic_mask):
        return []

    boxes = xywh_to_xyxy(prediction[dynamic_mask, :4])
    scores = scores[dynamic_mask]
    class_ids = class_ids[dynamic_mask]
    mask_coeffs = mask_coeffs[dynamic_mask]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_shape[0])

    keep = nms(boxes, scores, iou_thres)
    if keep.size == 0:
        return []

    detections = []
    for index in keep:
        detections.append(
            {
                "box": boxes[index],
                "score": float(scores[index]),
                "class_id": int(class_ids[index]),
                "mask_coeffs": mask_coeffs[index],
            }
        )
    return detections


def detection_mask_to_frame_mask(mask_coeffs, proto, box, meta, frame_shape, mask_threshold):
    """Project one detection mask back into original frame coordinates."""
    proto_h, proto_w = proto.shape[1:3]
    input_h, input_w = meta["input_shape"]
    original_h, original_w = frame_shape[:2]

    mask = sigmoid(mask_coeffs @ proto.reshape(proto.shape[0], -1))
    mask = mask.reshape(proto_h, proto_w)
    mask = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(input_w, x1))
    y1 = max(0, min(input_h, y1))
    x2 = max(0, min(input_w, x2))
    y2 = max(0, min(input_h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((original_h, original_w), dtype=np.uint8)

    boxed = np.zeros((input_h, input_w), dtype=np.float32)
    cropped = mask[y1:y2, x1:x2]
    boxed[y1:y2, x1:x2] = (cropped >= mask_threshold).astype(np.float32)

    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    unpad_w, unpad_h = meta["new_unpad"]
    boxed = boxed[pad_top : pad_top + unpad_h, pad_left : pad_left + unpad_w]
    boxed = cv2.resize(boxed, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return (boxed > 0).astype(np.uint8) * 255


def build_dynamic_mask(session, frame, input_size, conf_thres, iou_thres, mask_threshold, hole_close_kernel, dilate_kernel):
    """Run ONNX inference and combine all dynamic detections into one mask."""
    tensor, meta = preprocess(frame, input_size)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    prediction, proto = parse_outputs(outputs)

    detections = decode_detections(prediction, proto, meta["input_shape"], conf_thres, iou_thres)
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    if not detections:
        return combined

    for detection in detections:
        detection_mask = detection_mask_to_frame_mask(
            detection["mask_coeffs"],
            proto,
            detection["box"],
            meta,
            frame.shape,
            mask_threshold,
        )
        combined = np.maximum(combined, detection_mask)

    close_kernel = np.ones((hole_close_kernel, hole_close_kernel), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(combined)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    combined = filled

    dilate_kernel_matrix = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    combined = cv2.dilate(combined, dilate_kernel_matrix, iterations=1)
    return combined


def gray_out(frame, mask):
    """Replace masked (dynamic) pixels with a flat gray; keep background."""
    gray_fill = np.full_like(frame, 128)
    mask_3c = cv2.merge([mask, mask, mask])
    return np.where(mask_3c > 0, gray_fill, frame)


def build_session(model_path, providers):
    available = ort.get_available_providers()
    selected = []
    if providers:
        selected = [provider.strip() for provider in providers.split(",") if provider.strip()]
    else:
        preferred = ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        selected = [provider for provider in preferred if provider in available]
        if not selected and "CPUExecutionProvider" in available:
            selected = ["CPUExecutionProvider"]

    if not selected:
        raise RuntimeError(f"No usable ONNX Runtime providers found. Available providers: {available}")

    return ort.InferenceSession(str(model_path), providers=selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path to input frame (image)")
    parser.add_argument("--out", default="masked_frame.jpg", help="path to save output")
    parser.add_argument(
        "--model",
        default="yolov8n-seg.onnx",
        help="path to an exported YOLOv8 segmentation ONNX model",
    )
    parser.add_argument("--providers", default="", help="comma-separated ONNX Runtime providers")
    parser.add_argument("--input-size", type=int, default=640, help="ONNX input resolution")
    parser.add_argument("--hole-kernel", type=int, default=15, help="morph close kernel size")
    parser.add_argument("--dilate-kernel", type=int, default=7, help="edge safety-margin dilation size")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="mask binarization threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find ONNX model: {model_path}. Export your segmentation model to ONNX first."
        )

    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    session = build_session(model_path, args.providers)
    print(f"Using ONNX Runtime providers: {session.get_providers()}")

    mask = build_dynamic_mask(
        session,
        frame,
        input_size=args.input_size,
        conf_thres=args.conf,
        iou_thres=args.iou,
        mask_threshold=args.mask_thresh,
        hole_close_kernel=args.hole_kernel,
        dilate_kernel=args.dilate_kernel,
    )
    masked_frame = gray_out(frame, mask)

    cv2.imwrite(args.out, masked_frame)
    print(f"Saved masked frame -> {args.out}")


if __name__ == "__main__":
    main()