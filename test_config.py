# Test configuration với 200 ảnh và model pool đầy đủ

CUSTOM_PROBLEM = """
Bối cảnh của vấn đề:
Nhận dạng chữ viết tay là một bài toán cơ bản trong lĩnh vực học máy và xử lý ảnh, với nhiều ứng dụng thực tiễn như nhận dạng chữ số trên phiếu khảo sát, hóa đơn hay bài thi tự động. Việc phân loại chữ số viết tay thành số nguyên tố hoặc không giúp mở rộng khả năng ứng dụng trong các bài toán toán học tự động, kiểm tra bài tập, hoặc các ứng dụng giáo dục.

Yêu cầu cụ thể cần đạt được:
Xây dựng một hệ thống có khả năng nhận diện chữ số viết tay từ ảnh và xác định xem số đó có phải là số nguyên tố hay không.
Đầu vào là ảnh chứa một chữ số viết tay.
Đầu ra là nhãn phân loại nhị phân: "nguyên tố" hoặc "không nguyên tố".

Định dạng dữ liệu đầu vào cho bài toán tổng thể:
Một thư mục có tên là "images" chứa các ảnh grayscale 28x28 pixel, mỗi ảnh chứa một chữ số viết tay từ 0 đến 9.

Định dạng kết quả đầu ra mong muốn cho bài toán tổng thể:
File output.csv mỗi hàng là kết quả dự đoán mỗi ảnh
có các cột:
file_name: tên file ảnh
prediction: nhãn của ảnh:
"số nguyên tố" nếu số trong ảnh là số nguyên tố.
"không nguyên tố" nếu số trong ảnh không phải số nguyên tố.
"""

# Model pool đầy đủ theo table bạn cung cấp
CUSTOM_MODEL_POOL = [
    "https://huggingface.co/thanhtlx/image_classification_01",    # 1. Nhận diện ảnh chữ viết tay ⭐ QUAN TRỌNG NHẤT
    "https://huggingface.co/thanhtlx/text_classification_1",      # 2. Phân loại văn bản thành các topic
    "https://huggingface.co/zhaospei/Model_3",                   # 3. Dự đoán giá trung bình nhà ở các quận Cali
    "https://huggingface.co/zhaospei/Model_2",                   # 4. Dự đoán giá trung bình nhà ở các quận Cali  
    "https://huggingface.co/zhaospei/Model_4",                   # 5. Phân loại gian lận thẻ tín dụng
    "https://huggingface.co/zhaospei/Model_7",                   # 6. Phân loại hình ảnh ⭐ QUAN TRỌNG
    "https://huggingface.co/zhaospei/Model_6",                   # 7. Phân loại hình ảnh thành 100 lớp chi tiết ⭐ QUAN TRỌNG
]

DATASET_DRIVE_URL = "https://drive.google.com/drive/folders/1cijuTQmFaSew7IueB9eotOPYg1M90ppP"

CUSTOM_DATA_STRUCTURE = """
project/
├── images/              # Thư mục chứa ~200 ảnh PNG từ Google Drive
│   ├── 0.png           # Ảnh chữ số viết tay grayscale
│   ├── 1.png
│   ├── 2.png
│   ├── ...             # Có thể có đến 200 ảnh
│   ├── 198.png
│   └── 199.png
└── output.csv          # File output với ~200 predictions:
                        # file_name,prediction
                        # 0.png,số nguyên tố
                        # 1.png,không nguyên tố
                        # ...
"""

custom_test_config = {
    "task_description": "Digit Classification for Prime Number Detection",
    "problem_type": "Computer Vision - Image Classification",
    "dataset_info": {
        "type": "Image Dataset",
        "input_format": "PNG images 28x28 grayscale",
        "output_format": "CSV with predictions",
        "size": "~200 images",
        "source": "Google Drive handwritten digits"
    },
    "model_pool": [
        {
            "model_name": "thanhtlx/image_classification_01",
            "model_type": "Image Classification",
            "description": "Specialized handwritten digit classifier - BEST FOR THIS TASK",
            "huggingface_url": "https://huggingface.co/thanhtlx/image_classification_01",
            "priority": "HIGH - chuyên chữ viết tay"
        },
        {
            "model_name": "zhaospei/Model_6", 
            "model_type": "General CV Model",
            "description": "General computer vision model",
            "huggingface_url": "https://huggingface.co/zhaospei/Model_6",
            "priority": "MEDIUM - backup option"
        },
        {
            "model_name": "zhaospei/Model_7",
            "model_type": "General CV Model", 
            "description": "General computer vision model",
            "huggingface_url": "https://huggingface.co/zhaospei/Model_7",
            "priority": "MEDIUM - backup option"
        }
    ],
    "requirements": {
        "input": "Ảnh chữ số viết tay 28x28",
        "output": "CSV file: file_name, prediction (số nguyên tố / không nguyên tố)",
        "accuracy": "High precision cho digit recognition",
        "prime_numbers": [2, 3, 5, 7]
    },
    "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
    "constraints": {
        "time_limit": "10 minutes",
        "memory_limit": "8GB",
        "gpu_available": False
    },
    "expected_workflow": [
        "1. Load và preprocess images (28x28 grayscale)",
        "2. Use SINGLE BEST model cho digit classification", 
        "3. Convert digits to prime/non-prime labels",
        "4. Generate CSV output với predictions",
        "5. Validate results"
    ]
}