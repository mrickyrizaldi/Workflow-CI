import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load credential service account
creds = json.loads(os.environ["GDRIVE_CREDENTIALS"])
credentials = Credentials.from_service_account_info(
    creds,
    scopes=["https://www.googleapis.com/auth/drive"]
)

# Build Drive API
service = build('drive', 'v3', credentials=credentials)

# Gunakan ID Shared Drive (atau folder di Shared Drive) sebagai "parent"
SHARED_DRIVE_ID = os.environ["GDRIVE_FOLDER_ID"]

# FUNCTION: Upload folder rekursif
def upload_directory(local_dir_path, parent_drive_id):
    """
    Rekursif:
     - Jika item folder, buat folder di Drive, lalu panggil upload_directory lagi.
     - Jika item file, langsung upload ke parent_drive_id.
    """
    for item_name in os.listdir(local_dir_path):
        item_path = os.path.join(local_dir_path, item_name)
        if os.path.isdir(item_path):
            folder_meta = {
                'name': item_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_drive_id]
            }
            created_folder = service.files().create(
                body=folder_meta,
                fields='id',
                supportsAllDrives=True
            ).execute()
            new_folder_id = created_folder["id"]
            print(f"Created folder: {item_name} (ID: {new_folder_id})")

            # Rekursif ke subfolder
            upload_directory(item_path, new_folder_id)
        else:
            print(f"Uploading file: {item_name}")
            file_meta = {
                'name': item_name,
                'parents': [parent_drive_id]
            }
            media = MediaFileUpload(item_path, resumable=False)
            service.files().create(
                body=file_meta,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()

# MAIN UPLOAD PROCESS
print("STARTING UPLOAD ARTIFACT MLflow")
artifact_root = "./Mlflow-Artifact"

try:
    latest_folder = sorted(os.listdir(artifact_root))[-1]
except IndexError:
    raise RuntimeError("Tidak ditemukan folder artifact di Mlflow-Artifact!")

latest_mlruns_0_path = os.path.join(artifact_root, latest_folder, "mlruns", "0")
print(f"Latest artifact folder detected: {latest_folder}")

if not os.path.isdir(latest_mlruns_0_path):
    raise RuntimeError("Folder mlruns/0 tidak ditemukan")

for run_id in os.listdir(latest_mlruns_0_path):
    run_id_local_path = os.path.join(latest_mlruns_0_path, run_id)
    if run_id == "datasets":
        print("Skipping non-MLflow folder: datasets")
        continue

    if os.path.isdir(run_id_local_path):
        # Buat folder di Shared Drive
        run_id_folder_meta = {
            'name': f"{latest_folder}_{run_id}",
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [SHARED_DRIVE_ID]
        }
        run_id_folder = service.files().create(
            body=run_id_folder_meta,
            fields='id',
            supportsAllDrives=True
        ).execute()
        run_id_folder_id = run_id_folder["id"]
        print(f"Created run_id folder: {run_id} (ID: {run_id_folder_id})")

        # Upload rekursif isi foldernya
        upload_directory(run_id_local_path, run_id_folder_id)

print("Upload MLflow artifact selesai")
