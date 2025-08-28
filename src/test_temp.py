# Install once:
# pip install PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from ultralytics import YOLO
import cv2
from glob import glob
from pydrive import drive
import time
import psutil
import hashlib
import csv



# --- Authentication ---
gauth = GoogleAuth()

# Or load it manually:
BASE_DIR = os.path.dirname(__file__)
client_secret_path = os.path.join(BASE_DIR, 'src', 'client_secret.json')

# this line is used for absolute path
gauth.DEFAULT_SETTINGS['client_config_file'] = r'../model/client_secret.json'

#if u are working in colab use this line
#gauth.DEFAULT_SETTINGS['client_config_file'] = r'googledrive_path/model/client_secret.json'

# for colab access use these 2 line
#gauth = GoogleAuth()     #  use this for auth
#gauth.CommandLineAuth()  #  Use this instead of LocalWebserverAuth


gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# --- Configuration ---
# IDs of your Drive folders:
DRIVE_INPUT_FOLDER_ID = '1xezpPy1keNpYPxJM1gZ1SKmOobPCTlVx'
DRIVE_OUTPUT_FOLDER_ID = '1sigp9PQ9_uWKhrBTp673faPDD_vNpqXk'
TRAINNED_MODEL_FOLDER_ID = '1HF3Jr2hnOsx-_oaqoOuejFk36F3rKqHh'
MODEL_FILENAME = 'yolov5u.pt'







# --- Helper Functions ---

def download_model_from_drive(drive_instance, folder_id, model_name, local_save_dir):
    """
    Searches for and downloads a specific model file from a Google Drive folder.

    It first checks if the model already exists locally to avoid re-downloading.

    Args:
        drive_instance: An authenticated PyDrive GoogleDrive instance.
        folder_id (str): The ID of the Google Drive folder to search in.
        model_name (str): The exact name of the model file to download (e.g., 'best_v8l.pt').
        local_save_dir (str): The local directory where the model will be saved.

    Returns:
        str: The full local path to the downloaded model file, or None if not found.
    """
    os.makedirs(local_save_dir, exist_ok=True)
    local_model_path = os.path.join(local_save_dir, model_name)

    if os.path.exists(local_model_path):
        print(f"Model '{model_name}' already exists at '{local_model_path}'. Using local version.")
        return local_model_path

    print(f"Searching for model '{model_name}' in Google Drive folder ID: {folder_id}")
    query = f"'{folder_id}' in parents and title='{model_name}' and trashed=false"
    file_list = drive_instance.ListFile({'q': query}).GetList()

    if not file_list:
        print(f"Error: Model file '{model_name}' not found in the specified Drive folder.")
        return None

    model_file = file_list[0]
    print(f"Downloading '{model_file['title']}'...")
    model_file.GetContentFile(local_model_path)
    print(f"Model successfully downloaded to '{local_model_path}'")
    return local_model_path

def download_drive_folder(folder_id, local_folder, exts=('jpg','jpeg','png')):
    os.makedirs(local_folder, exist_ok=True)
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()
    for f in file_list:
        if any(f['title'].lower().endswith(ext) for ext in exts):
            print(f"Downloading {f['title']}…")
            f.GetContentFile(os.path.join(local_folder, f['title']))

def upload_to_drive(local_path, drive_folder_id):
    file_name = os.path.basename(local_path)
    gfile = drive.CreateFile({
        'parents': [{'id': drive_folder_id}],
        'title': file_name
    })
    gfile.SetContentFile(local_path)
    gfile.Upload()
    print(f"Uploaded {file_name}")





CSV_PATH = '../results/metrics.csv'
PLATFORM = 'Hailo Hat'            # or 'Local PC', 'GPU', etc.

CSV_FIELDS = [
    "image_na","image_sha","platform","model",
    "inference_ms","fps",
    "preprocess_ms","postprocess_ms",
    "cpu_percent","mem_percent","rss_mb",
    "detections","avg_confidence"
]

# one-time header
if not os.path.exists(os.path.dirname(CSV_PATH)):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _proc_stats():
    p = psutil.Process(os.getpid())
    # call once to initialize CPU calc, then instantaneous sample
    p.cpu_percent(None)
    cpu = p.cpu_percent(None)          # %
    mem_pct = p.memory_percent()       # %
    rss_mb = p.memory_info().rss / (1024*1024)
    return cpu, mem_pct, rss_mb




# --- Detection Functions ---

def detect_image(model, img_path, output_path=None):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img = cv2.resize(img, (720, 720))

    # run prediction (suppress console if you like with verbose=False)
    results = model.predict(img, verbose=False)
    r0 = results[0]

    annotated = r0.plot()
    if output_path:
        cv2.imwrite(output_path, annotated)
    return r0, output_path


def batch_detect_local(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    patterns = [os.path.join(input_dir, f'**/*.{ext}') for ext in ('jpg','jpeg','png')]
    files = sum([glob(p, recursive=True) for p in patterns], [])

    # model name from filename (e.g., yolov5su.pt -> yolov5su)
    try:
        model_name = os.path.splitext(os.path.basename(model.model.pt_path))[0]
    except Exception:
        model_name = os.path.basename(getattr(model, 'model', 'YOLO')).replace('.pt','')

    for img_path in files:
        name = os.path.basename(img_path)
        out = os.path.join(output_dir, name)
        print(f"→ {img_path} → {out}")

        # run detection & save annotated image
        r0, _ = detect_image(model, img_path, out)

        # Ultralytics speed (ms): 'preprocess', 'inference', 'postprocess'
        spd = getattr(r0, "speed", {}) or {}
        preprocess_ms  = float(spd.get("preprocess", 0.0))
        inference_ms   = float(spd.get("inference", 0.0))
        postprocess_ms = float(spd.get("postprocess", 0.0))

        # detections & confidence
        dets = 0
        avg_conf = 0.0
        try:
            dets = 0 if r0.boxes is None else len(r0.boxes)
            if dets > 0 and hasattr(r0.boxes, "conf") and r0.boxes.conf is not None:
                avg_conf = float(r0.boxes.conf.mean().item())
        except Exception:
            pass

        # FPS from inference time (ms)
        fps = (1000.0 / inference_ms) if inference_ms > 0 else 0.0

        # process stats
        cpu_pct, mem_pct, rss_mb = _proc_stats()

        # write CSV row
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                "image_na": name,
                "image_sha": _sha256(img_path),
                "platform": PLATFORM,
                "model": model_name,
                "inference_ms": round(inference_ms, 4),
                "fps": round(fps, 4),
                "preprocess_ms": round(preprocess_ms, 4),
                "postprocess_ms": round(postprocess_ms, 4),
                "cpu_percent": round(cpu_pct, 1),
                "mem_percent": round(mem_pct, 1),
                "rss_mb": round(rss_mb, 1),
                "detections": dets,
                "avg_confidence": round(avg_conf, 6),
            })

        # keep your Drive upload
        upload_to_drive(out, DRIVE_OUTPUT_FOLDER_ID)


# --- Main Execution ---
if __name__ == "__main__":
    # Define local directories

    # replace these to own loacl path
    LOCAL_IN = '../data/images/tmp_drive_in'
    LOCAL_OUT = '../results/images/tmp_drive_out'
    LOCAL_MODEL_DIR = '../model'

    # 1. Download the model from Google Drive
    model_path = download_model_from_drive(drive, TRAINNED_MODEL_FOLDER_ID, MODEL_FILENAME, LOCAL_MODEL_DIR)

    # 2. Check if model was downloaded and then load it
    if model_path:
        print("Loading YOLO model...")
        model = YOLO(model_path)
        print("Model loaded successfully.")

        # 3. Proceed with the detection workflow
        print("\n--- Starting Detection Workflow ---")
        download_drive_folder(DRIVE_INPUT_FOLDER_ID, LOCAL_IN)
        batch_detect_local(model, LOCAL_IN, LOCAL_OUT)
        print("\n--- Workflow Complete ---")
    else:
        print("Could not retrieve the model. Aborting operation.")
