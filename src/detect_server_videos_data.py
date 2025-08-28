# Install once:
# pip install PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from ultralytics import YOLO
import cv2
from glob import glob
from pydrive import drive

print("PWD:", os.getcwd())

# --- Authentication ---
gauth = GoogleAuth()
# Override default filename:
# Or load it manually:
BASE_DIR = os.path.dirname(__file__)
client_secret_path = os.path.join(BASE_DIR, 'src', 'client_secret.json')
# this line is used for absolute path
# gauth.DEFAULT_SETTINGS['client_config_file'] = r'C:\Users\iqbal\PycharmProjects\Capston_Project\src\client_secret.json'

gauth.DEFAULT_SETTINGS['client_config_file'] = r'../model/client_secret.json'

# for colab access use these 2 line
#gauth = GoogleAuth()     #  use this for auth
#gauth.CommandLineAuth()  #  Use this instead of LocalWebserverAuth


gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# --- Configuration ---
# IDs of your Drive folders:
DRIVE_INPUT_FOLDER_ID = '1EbM1lVtrB98aLRTFaT2xty-dZu2gZh3c'
DRIVE_OUTPUT_FOLDER_ID = '1Cg326PlRNoHyQvi0dEGhvlM7GpF1S9T7'
TRAINNED_MODEL_FOLDER_ID = '1HF3Jr2hnOsx-_oaqoOuejFk36F3rKqHh'
MODEL_FILENAME = 'best_v8l.pt'

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

def download_drive_folder(folder_id, local_folder, exts=('jpg','jpeg','png','mp4','avi','mov','mkv')):
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

# --- Detection Functions ---

def detect_image(model, img_path, output_path=None):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img = cv2.resize(img, (720, 720))
    results = model.predict(img)
    annotated = results[0].plot()
    if output_path:
        cv2.imwrite(output_path, annotated)
    return output_path

def batch_detect_local(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    patterns = [os.path.join(input_dir, f'**/*.{ext}') for ext in ('jpg','jpeg','png')]
    files = sum([glob(p, recursive=True) for p in patterns], [])
    for img_path in files:
        name = os.path.basename(img_path)
        out = os.path.join(output_dir, name)
        print(f"→ {img_path} → {out}")
        detect_image(model, img_path, out)
        upload_to_drive(out, DRIVE_OUTPUT_FOLDER_ID)



def batch_detect_videos(model, input_dir, output_dir=None, exts=('mp4', 'avi', 'mov', 'mkv')):
    """
    Runs detect_video on every file in input_dir matching exts.
    If output_dir is given, writes outputs there with the same basename.
    """
    # ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # gather all video paths
    patterns = [os.path.join(input_dir, f'**/*.{ext}') for ext in exts]
    video_files = []
    for pat in patterns:
        video_files.extend(glob(pat, recursive=True))

    for vid_path in video_files:
        name = os.path.basename(vid_path)
        # prepare output path if requested
        out_path = None
        if output_dir:
            out_path = os.path.join(output_dir, name)
        print(f"→ Processing {vid_path!r} → {out_path!r}")
        detect_video(model, vid_path, out_path)
        upload_to_drive(out_path, DRIVE_OUTPUT_FOLDER_ID)


def detect_video(model, video_path, output_path=None):
    """
    Run YOLOv8 inference on a video file (or webcam if video_path == 0).
    """
    cap = cv2.VideoCapture(video_path)
    # Optional: get codec and create VideoWriter if saving output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Inference
        results = model(frame)
        # Annotate frame
        annotated = results[0].plot()
        # Show
        cv2.imshow('YOLOv8 Video Detection', annotated)
        # Write to file
        if writer:
            writer.write(annotated)
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()



# --- Main Execution ---
if __name__ == "__main__":
    # Define local directories
    LOCAL_IN = '../data/videos/tmp_drive_in'
    LOCAL_OUT = '../results/videos/tmp_drive_out'
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
        batch_detect_videos(model, LOCAL_IN, LOCAL_OUT)
        #batch_detect_local(model, LOCAL_IN, LOCAL_OUT)
        print("\n--- Workflow Complete ---")
    else:
        print("Could not retrieve the model. Aborting operation.")
