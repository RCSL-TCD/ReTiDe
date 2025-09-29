import os
import shutil
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
def clean_results():
    if os.path.exists('processed'):
        shutil.rmtree('processed')
    os.makedirs('processed/images', exist_ok=True)
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')
    os.makedirs('uploads/images', exist_ok=True)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
