import zipfile
import os

def zip_project(output_filename, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {'.venv', '__pycache__'}

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Skip excluded dirs
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                filepath = os.path.join(root, file)
                if not any(excluded in filepath for excluded in exclude_dirs):
                    zipf.write(filepath, filepath)

zip_project("ArtistClassifier.zip")