import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
AGENTS_DIR = PROJECT_ROOT / "agents"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"

# Tạo directories nếu chưa có
for dir_path in [OUTPUTS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# General model pool cho different ML domains
DEFAULT_MODEL_POOL = [
    "https://huggingface.co/thanhtlx/image_classification_01",    # 1. Nhận diện ảnh chữ viết tay ⭐ QUAN TRỌNG NHẤT
    "https://huggingface.co/thanhtlx/text_classification_1",      # 2. Phân loại văn bản thành các topic
    "https://huggingface.co/zhaospei/Model_3",                   # 3. Dự đoán giá trung bình nhà ở các quận Cali
    "https://huggingface.co/zhaospei/Model_2",                   # 4. Dự đoán giá trung bình nhà ở các quận Cali  
    "https://huggingface.co/zhaospei/Model_4",                   # 5. Phân loại gian lận thẻ tín dụng
    "https://huggingface.co/zhaospei/Model_7",                   # 6. Phân loại hình ảnh ⭐ QUAN TRỌNG
    "https://huggingface.co/zhaospei/Model_6",                   # 7. Phân loại hình ảnh thành 100 lớp chi tiết ⭐ QUAN TRỌNG
]

# Default problem description
DEFAULT_PROBLEM = """
Analyze the provided data and create an appropriate machine learning solution.
The system should automatically detect the data type and apply suitable preprocessing,
model selection, and output formatting.
"""