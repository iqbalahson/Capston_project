import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import os
gauth = GoogleAuth()

BASE_DIR = os.path.dirname(__file__)
client_secret_path = os.path.join(BASE_DIR, 'src', 'client_secret.json')
# this line is used for absolute path
# gauth.DEFAULT_SETTINGS['client_config_file'] = r'/content/drive/MyDrive/Colab_Notebooks/Capston_Project-main/model/client_secret.json'

gauth.DEFAULT_SETTINGS['client_config_file'] = r'/content/drive/MyDrive/Colab_Notebooks/Capston_Project-main/model/client_secret.json'

#gauth.LocalWebserverAuth()

# Authenticate

gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

# Replace this with your actual folder ID
FOLDER_ID = '1sigp9PQ9_uWKhrBTp673faPDD_vNpqXk'

def delete_files_one_by_one(folder_id):
    print("Waiting 10 seconds...")
    time.sleep(10)
    while True:
        # List files in the folder
        file_list = drive.ListFile({
            'q': f"'{folder_id}' in parents and trashed=false"
        }).GetList()

        if not file_list:
            print("✅ Folder is empty.")
            break

        # Delete first file
        file = file_list[0]
        print(f"Deleting: {file['title']}")
        file.Delete()
        print("Deleted. Waiting 10 seconds...")
       

# Call the function
delete_files_one_by_one(FOLDER_ID)
