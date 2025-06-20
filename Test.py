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
Điều quan trọng là các công ty thẻ tín dụng có thể nhận ra các giao dịch thẻ tín dụng gian lận để khách hàng không phải trả tiền cho những mặt hàng mà họ không mua. Một vấn đề của task này là dữ liệu training khá là ít cho giao dịch gian lận gây vấn đề mất cân bằng nhãn dữ liệu.

Yêu cầu cụ thể cần đạt được:
Xây dựng một hệ thống có khả năng nhận đầu vào là một list các features liên quan đến sự gian lận của giao dịch và đầu ra hệ thống là phân loại giao dịch cụ thể đó có gian lận hay không
Mô hình cần xử lý đầu vào là 1 list các feature liên quan đến gian lận của giao dịch
Đầu ra là nhãn của giao dịch đó thuộc về

Định dạng dữ liệu đầu vào cho bài toán tổng thể:
Một file test.csv. Mỗi hàng là features cho từng task bao gồm các cột:
28 đặc trưng V1, V2, … V28 là các thành phần chính thu được từ PCA (Không được công bố cụ thể);  hai đặc trưng không được biến đổi bằng PCA là 'Time' và 'Amount'. Đặc trưng 'Time' thể hiện số giây đã trôi qua giữa mỗi giao dịch và giao dịch đầu tiên trong tập dữ liệu. Đặc trưng 'Amount' là số tiền của giao dịch, đặc trưng này có thể được sử dụng cho học tập nhạy cảm với chi phí phụ thuộc vào ví dụ và cuối cùng là ID: id của task
 
Định dạng kết quả đầu ra mong muốn cho bài toán tổng thể:
File output.csv mỗi hàng là kết quả dự đoán mỗi task
có các cột:
ID: id của task
class: 1 nếu là giao dịch gian lận, 0 với trường hợp ngược lại
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
DATA_SOURCE = "https://drive.google.com/drive/folders/1AuzgvKbGMi0XWBlryYkl_RPfUCKJXAZV?usp=sharing"

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