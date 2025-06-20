#!/usr/bin/env python3
"""
TEST CONFIGURATION
Sửa đổi config dưới đây để test các bài toán ML khác nhau
"""

import time
from main_pipeline import GeneralMLPipeline

# ============================================================================
# 🔧 TEST CONFIGURATION - CHỈNH SỬA PHẦN NÀY ĐỂ THAY ĐỔI TEST CASE
# ============================================================================

# 📝 PROBLEM DESCRIPTION - Mô tả bài toán cần giải quyết
PROBLEM_DESCRIPTION = """
"Bối cảnh của vấn đề:
Đây là một nhiệm vụ hồi quy. Mục tiêu là dự đoán giá trị trung bình của nhà ở các quận của California bằng cách sử dụng các đặc trưng từ cuộc điều tra dân số năm 1990, chẳng hạn như thu nhập trung bình, tuổi trung bình của nhà ở, tổng số phòng, v.v. ()

Yêu cầu cụ thể cần đạt được:
Xây dựng một hệ thống có khả năng nhận đầu vào là một list các features của khu vực có liên quan đến giá trung bình của nhà ở và đầu ra hệ thống là phân loại khu vực đó thuộc nhóm “rẻ”, “trung bình” hay “đắt” để dễ trực quan hóa hoặc định hướng chính sách
Mô hình cần xử lý đầu vào là 1 list các features liên quan đến giá trung bình của nhà ở
Đầu ra là nhãn của nhà ở khu vực đấy thuộc về

Định dạng dữ liệu đầu vào cho bài toán tổng thể:
Một file test.csv. Mỗi hàng là features cho từng task bao gồm các cột:

MedInc: Thu nhập trung vị của người dân sống trong khu vực (block)
HouseAge: Tuổi trung vị của các căn nhà trong khu vực
AveRooms: Số phòng trung bình của các căn nhà trong khu vực
AveBedrms: Số phòng ngủ trung bình của các căn nhà trong khu vực
Population: Số người sinh sống trong khu vực
AveOccup: Số người trung bình sống chung trong một hộ (dưới cùng một mái nhà)
Latitude: Vĩ độ địa lý
Longitude: Kinh độ địa lý
ID: id của task
 
Định dạng kết quả đầu ra mong muốn cho bài toán tổng thể:
File output.csv mỗi hàng là kết quả dự đoán mỗi task
có các cột:
ID: id của task
MedHouseVal: nhóm  nhà ở khu vực đó thuộc về

Các nhóm có thể là:
low: nhóm ""rẻ"" có giá trung bình < 1
medium: nhóm ""rẻ"" có 1 <= giá trung bình < 2.5
high: nhóm ""rẻ"" có giá trung bình >= 2.5

"
"""

# 🤖 MODEL POOL - Danh sách các model HuggingFace để lựa chọn
MODEL_POOL = [
    "https://huggingface.co/thanhtlx/image_classification_01",
    "https://huggingface.co/zhaospei/Model_3",
    "https://huggingface.co/zhaospei/Model_2", 
    "https://huggingface.co/zhaospei/Model_4",
    "https://huggingface.co/zhaospei/Model_7",
    "https://huggingface.co/zhaospei/Model_6",
    "https://huggingface.co/thanhtlx/text_classification_1",
]

# 🔗 DATA SOURCE - Link Google Drive hoặc đường dẫn dữ liệu
DATA_SOURCE = "https://drive.google.com/drive/folders/1wdhKcJGNqGF-CXopxIFygrojb1gR31xI?usp=sharing"

# 📊 TEST INFO - Thông tin mô tả test case

# ============================================================================
# 💡 EXAMPLE CONFIGS - Uncomment để sử dụng các test case khác
# ============================================================================

# # 📰 NEWS CLASSIFICATION CONFIG:
# PROBLEM_DESCRIPTION = """
# Bối cảnh của vấn đề:
# Trong thời đại thông tin hiện nay, người dùng phải tiếp nhận một lượng lớn tin tức mỗi ngày từ nhiều nguồn khác nhau. Việc tự động phân loại các đoạn tin tức theo chủ đề giúp hệ thống quản lý nội dung hiệu quả hơn, đồng thời hỗ trợ người dùng tìm kiếm, lọc, và tiếp cận thông tin theo lĩnh vực quan tâm.

# Yêu cầu cụ thể cần đạt được:
# Xây dựng một ứng dụng có khả năng phân loại đoạn văn bản tiếng Anh mô tả nội dung một bản tin vào đúng nhóm chủ đề tương ứng.

# Định dạng dữ liệu đầu vào: Một đoạn văn bản ngắn bằng tiếng Anh, có nội dung mô tả một bản tin.
# Định dạng kết quả đầu ra: Một nhãn chủ đề duy nhất cho mỗi đoạn tin tức đầu vào.
# Các nhãn thuộc tập sau: 'Marketplace', 'Recreation', 'Technology', 'Politics', 'Religion'
# """
# MODEL_POOL = [
#     "https://huggingface.co/thanhtlx/text_classification_1",
#     "https://huggingface.co/zhaospei/Model_3",
#     "https://huggingface.co/zhaospei/Model_2",
#     "https://huggingface.co/zhaospei/Model_4",
#     "https://huggingface.co/zhaospei/Model_7",
#     "https://huggingface.co/zhaospei/Model_6",
#     "https://huggingface.co/thanhtlx/image_classification_01",
# ]
# DATA_SOURCE = "https://drive.google.com/drive/folders/1yrSszfKlU8Tk9tySPxlA_tbPya9Wi7uQ"
# TEST_NAME = "News Classification"
# TEST_DESCRIPTION = "Phân loại tin tức tiếng Anh thành 5 categories"
# EXPECTED_OUTPUT = "Predictions với categories: Marketplace, Recreation, Technology, Politics, Religion"

# # 🖼️ IMAGE CLASSIFICATION CONFIG:
# PROBLEM_DESCRIPTION = """
# Phân loại ảnh thành các categories khác nhau sử dụng computer vision.
# 
# Yêu cầu cụ thể:
# - Nhận diện và phân loại objects/scenes trong ảnh
# - Dự đoán category chính của mỗi ảnh
# - Xử lý các định dạng ảnh phổ biến (JPG, PNG)
# 
# Định dạng đầu vào: Ảnh digital trong các format phổ biến
# Định dạng đầu ra: CSV với filename và predicted category
# """
# MODEL_POOL = [
#     "https://huggingface.co/thanhtlx/image_classification_01",
#     "https://huggingface.co/zhaospei/Model_3",
#     "https://huggingface.co/zhaospei/Model_2",
# ]
# DATA_SOURCE = "https://drive.google.com/drive/folders/your_image_folder_id"
# TEST_NAME = "Image Classification"
# TEST_DESCRIPTION = "Phân loại ảnh thành các categories"
# EXPECTED_OUTPUT = "CSV với filename và predicted categories"

# ============================================================================
# 🚀 PIPELINE EXECUTION - KHÔNG CẦN SỬA PHẦN NÀY
# ============================================================================

def run_ml_pipeline():

    print(f"🤖 Model Pool: {len(MODEL_POOL)} models available")
    print(f"🔗 Data Source: {DATA_SOURCE[:60]}...")
    print("="*80)
    
    # Khởi tạo pipeline
    pipeline = GeneralMLPipeline()
    
    # Chạy pipeline
    start_time = time.time()
    try:
        print("🚀 Executing ML pipeline...")
        result = pipeline.execute_pipeline(
            problem_description=PROBLEM_DESCRIPTION,
            model_pool=MODEL_POOL,
            data_source=DATA_SOURCE
        )
        
        execution_time = time.time() - start_time
        
        print("="*60)
        print(f"✅ Success: {result.success}")
        print(f"📄 Generated Script: {result.output_file}")
        print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
        
        if result.success:
            print(f"\n📊 Pipeline Statistics:")
            print(f"   Selected Models: {result.statistics.get('selected_models', 0)}")
            print(f"   Data Source: {result.statistics.get('data_source', 'N/A')[:50]}...")
            print(f"   Problem Type: {result.statistics.get('problem_type', 'N/A')[:50]}...")
            
            print(f"\n🎯 Next Steps:")
            print(f"   1. Run the generated script: python {result.output_file}")
            print(f"   2. Check output files for ML results")

        
        return result.success
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ PIPELINE FAILED: {e}")
        print(f"🐛 Error Details: {type(e).__name__}")
        print(f"⏱️ Time Before Failure: {execution_time:.2f} seconds")
        
        import traceback
        traceback.print_exc()
        
        return False

def main():
    """Main execution"""
    print("🎯 CONFIGURABLE ML PIPELINE TEST")
    print("="*80)
    print("💡 To change test case:")
    print("   1. Edit PROBLEM_DESCRIPTION in config section")
    print("   2. Edit MODEL_POOL list")
    print("   3. Edit DATA_SOURCE link")
    print("   4. Uncomment example configs if needed")
    print("="*80)
    
    print(f"\n📋 CURRENT CONFIG:")
    print(f"   🤖 Models: {len(MODEL_POOL)} options")
    print(f"   🔗 Data: {DATA_SOURCE[:50]}...")
    
    # Confirm before running
    print(f"\n🚀 Ready to run with current config? (y/n)")
    choice = input("Choice: ").strip().lower()
    
    if choice != 'y':
        print("❌ Execution cancelled. Edit config section and run again.")
        return
    
    # Run the pipeline
    success = run_ml_pipeline()
    
    print(f"\n🏁 FINAL RESULT:")
    if success:
        print("🎉 THÀNH CÔNG! General pipeline đã tạo solution cho ML task!")
        print("📝 Check outputs/ folder cho generated script")
    else:
        print("❌ THẤT BẠI! Xem error logs ở trên để debug")
    
    print("="*80)
    print("💡 To test other problems, edit the config section and run again!")

if __name__ == "__main__":
    main() 