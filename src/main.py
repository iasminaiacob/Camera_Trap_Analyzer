import os
import io
import json
import zipfile
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2
import numpy as np
import requests
from tqdm import tqdm

AZURE_BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"
META_ZIP_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def download_bytes(url: str, timeout: int = 60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def load_coco_camera_traps(meta_zip_url: str, meta_dir: str) -> Dict:
    """
    Downloads and extracts COCO Camera Traps JSON zip, loads the JSON into memory.
    """
    ensure_dir(meta_dir)
    zip_path = os.path.join(meta_dir, "caltech_camera_traps.json.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading metadata zip...\n  {meta_zip_url}")
        data = download_bytes(meta_zip_url)
        with open(zip_path, "wb") as f:
            f.write(data)

    #extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        #find the first .json inside
        json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not json_names:
            raise RuntimeError("No .json found inside metadata zip.")
        json_name = json_names[0]
        extracted_json_path = os.path.join(meta_dir, os.path.basename(json_name))

        if not os.path.exists(extracted_json_path):
            print(f"Extracting {json_name} ...")
            zf.extract(json_name, meta_dir)
            #if extracted inside a subfolder, move it up
            src_path = os.path.join(meta_dir, json_name)
            if src_path != extracted_json_path:
                os.replace(src_path, extracted_json_path)

    print(f"Loading metadata JSON:\n  {extracted_json_path}")
    with open(extracted_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("Example image keys:", list(meta["images"][0].keys()))
    print("Example file_name:", meta["images"][0].get("file_name"))


    return meta

def build_indices(meta: Dict):
    """
    Builds helpful lookups:
    - cat_id -> cat_name
    - image_id -> image_info
    - image_id -> list of annotation category_ids
    Also groups images by (location, seq_id) if present.
    """
    cat_id_to_name = {c["id"]: c["name"] for c in meta.get("categories", [])}
    images = meta.get("images", [])
    anns = meta.get("annotations", [])

    image_id_to_info = {img["id"]: img for img in images}

    image_id_to_cat_ids = defaultdict(list)
    for a in anns:
        image_id_to_cat_ids[a["image_id"]].append(a["category_id"])

    seq_map = defaultdict(list)  #key -> list[image_id]
    for img in images:
        loc = img.get("location", None)
        seq = img.get("seq_id", None)

        if loc is not None and seq is not None:
            key = (str(loc), str(seq))
        else:
            #fallback: group by parent folder of file_name
            fn = img.get("file_name", "")
            key = (os.path.dirname(fn), "NA")

        seq_map[key].append(img["id"])

    #sort each sequence by frame number if available; else by file_name
    for key, ids in seq_map.items():
        def sort_key(image_id):
            info = image_id_to_info[image_id]
            if "frame_num" in info:
                return info["frame_num"]
            return info.get("file_name", "")
        ids.sort(key=sort_key)

    return cat_id_to_name, image_id_to_info, image_id_to_cat_ids, seq_map

def gt_is_animal_present(image_id: int, image_id_to_cat_ids: Dict[int, List[int]], cat_id_to_name: Dict[int, str]) -> int:
    """
    Ground truth binary label: 1 = animal present, 0 = empty
    If any annotation category is not 'empty', we count it as animal present.
    """
    cat_ids = image_id_to_cat_ids.get(image_id, [])
    if not cat_ids:
        #some images might have no annotations. we treat them as empty.
        return 0

    names = [cat_id_to_name.get(cid, "") for cid in cat_ids]
    #if all are 'empty', then empty. otherwise animal present
    return 0 if all(n == "empty" for n in names) else 1

def fetch_image_cached(file_name: str, cache_dir: str) -> np.ndarray:
    """
    Downloads one image from Azure base URL if not cached; returns BGR image.
    """
    ensure_dir(cache_dir)

    safe_rel = file_name.replace("/", os.sep)
    local_path = os.path.join(cache_dir, safe_rel)

    ensure_dir(os.path.dirname(local_path))

    if not os.path.exists(local_path):
        url = f"{AZURE_BASE}/{file_name}"
        img_bytes = download_bytes(url, timeout=120)
        with open(local_path, "wb") as f:
            f.write(img_bytes)

    img = cv2.imread(local_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read cached image: {local_path}")
    return img

def median_background(frames_gray: List[np.ndarray]) -> np.ndarray:
    """
    Per-pixel median across frames
    """
    stack = np.stack(frames_gray, axis=0).astype(np.uint8)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg

def detect_foreground_mask(gray: np.ndarray, bg: np.ndarray, diff_thresh: int = 25) -> np.ndarray:
    """
    Foreground by abs difference + threshold + morphology
    """
    diff = cv2.absdiff(gray, bg)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    #remove small noise, fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def classify_animal_present(mask: np.ndarray, area_thresh: int = 1500) -> Tuple[int, int]:
    """
    Returns (pred_label, foreground_area)
    pred_label: 1 if animal present. else 0
    """
    fg_area = int(np.count_nonzero(mask))
    pred = 1 if fg_area >= area_thresh else 0
    return pred, fg_area

def draw_overlay(bgr: np.ndarray, mask: np.ndarray, pred: int, fg_area: int, gt: int) -> np.ndarray:
    """
    Overlays mask on image and writes text
    """
    overlay = bgr.copy()
    mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # tint mask in red
    mask_col[:, :, 0] = 0
    mask_col[:, :, 1] = 0
    overlay = cv2.addWeighted(overlay, 1.0, mask_col, 0.35, 0)

    txt = f"pred={pred} gt={gt} fg_area={fg_area}"
    color = (0, 255, 0) if pred == gt else (0, 0, 255)
    cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return overlay

def run_experiment(
    meta_zip_url: str,
    cache_dir: str,
    meta_dir: str,
    out_dir: str,
    max_sequences: int = 30,
    frames_per_seq: int = 8,
    diff_thresh: int = 25,
    area_thresh: int = 1500,
    random_seed: int = 42,
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "overlays"))
    ensure_dir(os.path.join(out_dir, "reports"))

    meta = load_coco_camera_traps(meta_zip_url, meta_dir)
    cat_id_to_name, image_id_to_info, image_id_to_cat_ids, seq_map = build_indices(meta)

    keys = list(seq_map.keys())
    #keep only sequences with enough frames
    keys = [k for k in keys if len(seq_map[k]) >= frames_per_seq]
    random.Random(random_seed).shuffle(keys)
    keys = keys[:max_sequences]

    results = []
    TP = FP = TN = FN = 0
    correct = 0
    total = 0

    for seq_idx, key in enumerate(tqdm(keys, desc="Processing sequences")):
        image_ids = seq_map[key][:frames_per_seq]  #only first N frames in sequence
        frames_gray = []
        frames_bgr = []
        file_names = []

        #download and load frames
        for image_id in image_ids:
            info = image_id_to_info[image_id]
            file_name = info["file_name"]
            file_names.append(file_name)

            bgr = fetch_image_cached(file_name, cache_dir)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            frames_bgr.append(bgr)
            frames_gray.append(gray)

        #build background model
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=25,
            detectShadows=False
        )

        #let MOG2 learn the background
        mog2.apply(frames_gray[0], learningRate=1.0)

        #evaluate each frame in sequence
        for k_frame, image_id in enumerate(image_ids):
            if k_frame == 0:
                continue #skip first frame (used for background model)
            info = image_id_to_info[image_id]
            file_name = info["file_name"]

            #animal present=1; empty=0
            gt = gt_is_animal_present(image_id, image_id_to_cat_ids, cat_id_to_name)

            #foreground mask
            mask = mog2.apply(frames_gray[k_frame], learningRate=0)
            mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

            #prediction
            pred, fg_area = classify_animal_present(mask, area_thresh=area_thresh)

            #update confusion matrix
            if pred == 1 and gt == 1:
                TP += 1
            elif pred == 1 and gt == 0:
                FP += 1
            elif pred == 0 and gt == 0:
                TN += 1
            elif pred == 0 and gt == 1:
                FN += 1

            #accuracy counters
            correct += int(pred == gt)
            total += 1

            #save results
            results.append({
                "seq_key": str(key),
                "image_id": image_id,
                "file_name": file_name,
                "pred": pred,
                "gt": gt,
                "fg_area": fg_area
            })

            #save overlay img
            overlay = draw_overlay(frames_bgr[k_frame], mask, pred, fg_area, gt)
            out_name = f"seq{seq_idx:03d}_f{k_frame:02d}_pred{pred}_gt{gt}.jpg"
            cv2.imwrite(os.path.join(out_dir, "overlays", out_name), overlay)


    acc = correct / max(total, 1)
    print(f"\nDone. Frames processed: {total}, accuracy: {acc:.3f}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print("\nConfusion Matrix: ")
    print(f"TP: {TP}, FP: {FP}")
    print(f"FN: {FN}, TN: {TN}")

    print(f"\nPrecision: {precision:.3f}")
    print(f"\nRecall: {recall:.3f}")

    #save CSV report
    csv_path = os.path.join(out_dir, "reports", "results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("seq_key,image_id,file_name,pred,gt,fg_area\n")
        for r in results:
            f.write(f"{r['seq_key']},{r['image_id']},{r['file_name']},{r['pred']},{r['gt']},{r['fg_area']}\n")

    print(f"Saved overlays to: {os.path.join(out_dir, 'overlays')}")
    print(f"Saved report to:   {csv_path}")

if __name__ == "__main__":
    run_experiment(
        meta_zip_url=META_ZIP_URL,
        cache_dir="Camera_Trap_Analyzer/data/cache",
        meta_dir="Camera_Trap_Analyzer/data/meta",
        out_dir="Camera_Trap_Analyzer/outputs",
        max_sequences=30, #sequences to sample
        frames_per_seq=3,
        diff_thresh=20, #pixel difference threshold
        area_thresh=300, #foreground area threshold
        random_seed=42
    )