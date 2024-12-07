from flask import Flask, request, render_template, send_file
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# 獲取程式檔案的根目錄，並設置相對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded.", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected.", 400

    # 保存上傳的圖片
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # 接收用戶輸入的 n_components
    n_components = int(request.form.get('components', 50))  # 默認為 50
    print(f"User-specified n_components: {n_components}")

    # 圖片降維處理
    reduced_path = process_image(filepath, n_components)

    # 直接返回處理後的文件，觸發下載
    return send_file(reduced_path, as_attachment=True, download_name=f'reduced_{file.filename}')

def process_image(filepath, n_components):
    try:
        img = Image.open(filepath)
        img_data = np.array(img)

        print(f"Original image shape: {img_data.shape}")

        # 如果圖片是彩色，對每個通道分別降維
        if img_data.ndim == 3 and img_data.shape[2] == 3:
            r, g, b = img_data[:, :, 0], img_data[:, :, 1], img_data[:, :, 2]

            print(f"Reducing each channel with n_components={n_components}")
            pca = PCA(n_components=n_components)

            r_transformed = pca.fit_transform(r)
            r_reconstructed = pca.inverse_transform(r_transformed)

            g_transformed = pca.fit_transform(g)
            g_reconstructed = pca.inverse_transform(g_transformed)

            b_transformed = pca.fit_transform(b)
            b_reconstructed = pca.inverse_transform(b_transformed)

            reconstructed_data = np.stack((r_reconstructed, g_reconstructed, b_reconstructed), axis=2)
        else:
            # 灰階圖片處理
            print("Image is grayscale.")
            flat_data = img_data
            pca = PCA(n_components=n_components)
            pca_transformed = pca.fit_transform(flat_data)
            reconstructed_data = pca.inverse_transform(pca_transformed)

        reconstructed_data = np.clip(reconstructed_data, 0, 255)

        reduced_filename = f'reduced_{os.path.basename(filepath)}'
        reduced_path = os.path.join(RESULT_FOLDER, reduced_filename)
        reconstructed_image = Image.fromarray(reconstructed_data.astype(np.uint8))
        reconstructed_image.save(reduced_path)

        print(f"Reduced image saved at: {reduced_path}")
        return reduced_path
    except Exception as e:
        print(f"Error during image processing: {e}")
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)