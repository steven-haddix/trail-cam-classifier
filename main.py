from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path


import cv2
import numpy as np

load_dotenv()

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
DEFAULT_WEIGHTS = (0.45, 0.45, 0.1)


@dataclass
class Candidate:
    frame_index: int
    time_s: float
    sharpness: float
    brightness: float
    motion: float
    hash_bits: np.ndarray
    score: float = 0.0


def parse_weights(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "weights must be three comma-separated numbers: sharpness,motion,brightness"
        )
    try:
        weights = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be numeric values") from exc
    total = sum(weights)
    if total <= 0:
        return DEFAULT_WEIGHTS
    return tuple(weight / total for weight in weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract best frames from trail cam footage for LLM classification."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="video-samples",
        help="Video file or directory to scan.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="extracted-frames",
        help="Directory to write selected frames.",
    )
    parser.add_argument(
        "--per-video",
        "-k",
        type=int,
        default=8,
        help="Number of frames to save per video (or per tile if --tile is set).",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Generate a 2x2 grid tile of the best frames.",
    )
    parser.add_argument(
        "--min-dist-frames",
        type=int,
        default=15,
        help="Minimum number of frames between selected candidates.",
    )
    parser.add_argument(
        "--use-mog2",
        action="store_true",
        default=True,
        help="Use MOG2 background subtractor for motion scoring.",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=0.5,
        help="Seconds between sampled frames.",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=640,
        help="Resize width for scoring. Use 0 to disable resizing.",
    )
    parser.add_argument(
        "--save-max-width",
        type=int,
        default=0,
        help="Resize width for saved frames. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for saved frames (0-100).",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Minimum sharpness to keep a frame.",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.0,
        help="Minimum motion ratio (0-1) to keep a frame.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=15,
        help="Pixel difference threshold used for motion detection.",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=int,
        default=5,
        help="Max Hamming distance for aHash deduplication (0 disables).",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=DEFAULT_WEIGHTS,
        help="Comma-separated weights: sharpness,motion,brightness.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Classify hero tiles using Google Gemini (requires GEMINI_API_KEY).",
    )
    args = parser.parse_args()
    if args.sample_every <= 0:
        parser.error("--sample-every must be greater than 0")
    if args.per_video <= 0:
        parser.error("--per-video must be greater than 0")
    return args


def list_video_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS
            ]
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    new_height = max(1, int(round(height * scale)))
    new_size = (max_width, new_height)
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def average_hash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    return (small > avg).astype(np.uint8).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        return [0.0 for _ in values]
    scale = max_val - min_val
    return [(value - min_val) / scale for value in values]


def analyze_frame(
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    analysis_width: int,
    motion_threshold: int,
    fgbg: cv2.BackgroundSubtractorMOG2 | None = None,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    resized = resize_max_width(frame, analysis_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    if fgbg is not None:
        # MOG2 handles lighting shifts and tree-sway better than simple differencing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = fgbg.apply(blurred)
        motion = float(np.mean(fg_mask > 0))
    elif prev_gray is None:
        motion = 0.0
    else:
        diff = cv2.absdiff(gray, prev_gray)
        motion = float(np.mean(diff > motion_threshold))

    hash_bits = average_hash(gray)
    return gray, sharpness, brightness, motion, hash_bits


def score_candidates(
    candidates: list[Candidate], weights: tuple[float, float, float]
) -> None:
    sharpness_values = [candidate.sharpness for candidate in candidates]
    motion_values = [candidate.motion for candidate in candidates]
    brightness_values = [candidate.brightness for candidate in candidates]
    sharpness_norm = normalize(sharpness_values)
    motion_norm = normalize(motion_values)
    brightness_norm = normalize(brightness_values)
    for candidate, sharp_norm, motion_norm, bright_norm in zip(
        candidates, sharpness_norm, motion_norm, brightness_norm
    ):
        brightness_score = max(0.0, 1.0 - abs(bright_norm - 0.5) * 2.0)
        candidate.score = (
            weights[0] * sharp_norm
            + weights[1] * motion_norm
            + weights[2] * brightness_score
        )


def select_candidates(
    candidates: list[Candidate],
    top_k: int,
    dedupe_threshold: int,
    min_dist_frames: int = 0,
) -> list[Candidate]:
    sorted_candidates = sorted(
        candidates, key=lambda candidate: candidate.score, reverse=True
    )
    selected: list[Candidate] = []
    for candidate in sorted_candidates:
        if len(selected) >= top_k:
            break

        # Temporal distance filter
        if min_dist_frames > 0 and any(
            abs(candidate.frame_index - kept.frame_index) < min_dist_frames
            for kept in selected
        ):
            continue

        # Perceptual hash deduplication
        if dedupe_threshold > 0 and any(
            hamming_distance(candidate.hash_bits, kept.hash_bits) <= dedupe_threshold
            for kept in selected
        ):
            continue

        selected.append(candidate)

    # Fallback: if we didn't get enough candidates, relax the constraints
    if len(selected) < top_k:
        for candidate in sorted_candidates:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= top_k:
                break

    # Final sort by frame index to keep them chronological if desired,
    # but usually we want them by score for the tile.
    # Let's keep them in the order they were selected (best first).
    return selected


def create_vlm_tile(
    frames: list[np.ndarray], cols: int = 2, rows: int = 2
) -> np.ndarray:
    """
    Tiles frames into a grid for efficient LLM token usage.
    """
    if not frames:
        return np.array([], dtype=np.uint8)

    # Use the first frame to determine standardized tile size
    h, w = frames[0].shape[:2]

    # Create empty canvas
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        if i >= rows * cols:
            break
        r = i // cols
        c = i % cols

        # Resize if necessary to match the first frame's dimensions
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame

    return canvas


def save_selected_frames(
    video_path: Path,
    output_dir: Path,
    selected: list[Candidate],
    fps: float,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping {video_path.name}: cannot reopen video for saving.")
        return
    metadata = {
        "video": video_path.name,
        "fps": round(fps, 3),
        "sample_every_s": args.sample_every,
        "per_video": args.per_video,
        "frames": [],
    }
    saved_frames = []
    for rank, candidate in enumerate(selected, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
        ok, frame = cap.read()
        if not ok:
            print(
                f"Failed to read frame {candidate.frame_index} from {video_path.name}."
            )
            continue
        frame = resize_max_width(frame, args.save_max_width)
        saved_frames.append((rank, candidate, frame))

    if args.tile and saved_frames:
        # Use top 4 for 2x2 grid
        tile_frames = [f for r, c, f in saved_frames[:4]]
        tile = create_vlm_tile(tile_frames, cols=2, rows=2)
        tile_path = output_dir / "hero_tile.jpg"
        cv2.imwrite(
            str(tile_path),
            tile,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        metadata["tile"] = tile_path.name

    for rank, candidate, frame in saved_frames:
        filename = f"rank_{rank:02d}_frame_{candidate.frame_index:06d}.jpg"
        out_path = output_dir / filename
        success = cv2.imwrite(
            str(out_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        if not success:
            print(f"Failed to save {out_path}.")
            continue
        metadata["frames"].append(
            {
                "rank": rank,
                "frame_index": candidate.frame_index,
                "time_s": round(candidate.time_s, 3),
                "score": round(candidate.score, 6),
                "sharpness": round(candidate.sharpness, 3),
                "motion": round(candidate.motion, 6),
                "brightness": round(candidate.brightness, 3),
                "file": out_path.name,
            }
        )
    cap.release()
    if args.classify and "tile" in metadata:
        import os

        from classifier import classify_tile

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(
                f"Skipping classification for {video_path.name}: GEMINI_API_KEY not set."
            )
        else:
            print(f"Classifying {metadata['tile']} for {video_path.name}...")
            tile_path = output_dir / metadata["tile"]
            try:
                result = classify_tile(tile_path, api_key=api_key)
                metadata["classification"] = result.model_dump()
                print(f"Classification summary: {result.summary}")
            except Exception as e:
                print(f"Classification failed for {video_path.name}: {e}")

    (output_dir / "index.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"{video_path.name}: saved {len(metadata['frames'])} frames to {output_dir}")


def process_video(
    video_path: Path, output_root: Path, args: argparse.Namespace
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping {video_path.name}: cannot open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or math.isnan(fps):
        fps = 30.0
    step = max(int(round(args.sample_every * fps)), 1)
    candidates: list[Candidate] = []
    prev_gray = None
    frame_index = 0

    fgbg = None
    if args.use_mog2:
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )

    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray, sharpness, brightness, motion, hash_bits = analyze_frame(
                frame, prev_gray, args.analysis_width, args.motion_threshold, fgbg=fgbg
            )
            if sharpness >= args.min_sharpness and motion >= args.min_motion:
                candidates.append(
                    Candidate(
                        frame_index=frame_index,
                        time_s=frame_index / fps,
                        sharpness=sharpness,
                        brightness=brightness,
                        motion=motion,
                        hash_bits=hash_bits,
                    )
                )
            prev_gray = gray
        frame_index += 1
    cap.release()
    if not candidates:
        print(f"No candidate frames found for {video_path.name}.")
        return
    score_candidates(candidates, args.weights)
    selected = select_candidates(
        candidates,
        args.per_video,
        args.dedupe_threshold,
        min_dist_frames=args.min_dist_frames,
    )
    save_selected_frames(video_path, output_root / video_path.stem, selected, fps, args)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)
    try:
        video_paths = list_video_files(input_path)
    except FileNotFoundError as exc:
        print(str(exc))
        return
    if not video_paths:
        print(f"No videos found in {input_path}")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    for video_path in video_paths:
        process_video(video_path, output_root, args)


if __name__ == "__main__":
    main()
