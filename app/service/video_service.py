
from fastapi import File, UploadFile
from google.cloud import storage
import uuid

bucket_name = 'entro-tad-videos'
bucket_folder = 'raw-uploads/'
local_folder = 'local-videos'


storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)

def create_upload_video(video: UploadFile = File(...)):
    destination_blob_name = f'{uuid.uuid4()}-{video.filename}'
    blob = bucket.blob(bucket_folder + destination_blob_name)
    blob.upload_from_file(video.file)
    return f'Uploaded {video.filename} to "{bucket_name}" bucket.'

def list_videos():
    files = bucket.list_blobs(prefix=bucket_folder)
    fileList = [file.name for file in files if '.' in file.name]
    return fileList
