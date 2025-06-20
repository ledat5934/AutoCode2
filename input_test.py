import os
import pandas as pd
from typing import List, Tuple
import time
from main_pipeline import GeneralMLPipeline
def describe_dataset(root_dir: str,
                     max_files_per_dir: int = 3,
                     sample_rows: int = 5) -> Tuple[str, str]:
    """
    Trả về:
      • description  – nội dung của file description.txt (nếu tồn tại, else "")
      • summary_str  – chuỗi mô tả cây thư mục & tóm tắt CSV
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"'{root_dir}' không phải thư mục hợp lệ.")

    # ───── 1) Đọc description.txt ─────
    desc_file = os.path.join(root_dir, "description.txt")
    description = ""
    if os.path.isfile(desc_file):
        with open(desc_file, encoding="utf-8") as f:
            description = f.read()

    # ───── 2) Xây chuỗi summary_str (như phần trước) ─────
    lines: List[str] = []
    root_name = os.path.basename(os.path.normpath(root_dir))
    lines.append(f"Cấu trúc thư mục dữ liệu {root_dir}:")

    csv_paths: List[str] = []

    for dirpath, _, filenames in os.walk(root_dir):
        filenames.sort()
        rel_dir = os.path.relpath(dirpath, root_dir)
        rel_dir = "" if rel_dir == "." else rel_dir

        for fname in filenames[:max_files_per_dir]:
            rel_file = os.path.join(rel_dir, fname) if rel_dir else fname
            lines.append(rel_file)

            if fname.lower().endswith(".csv"):
                csv_paths.append(rel_file)

        if len(filenames) > max_files_per_dir:
            prefix = f"{rel_dir}\\" if rel_dir else ""
            lines.append(f"{prefix}...")

        # thu thập .csv vượt giới hạn
        for fname in filenames[max_files_per_dir:]:
            if fname.lower().endswith(".csv"):
                rel_file = os.path.join(rel_dir, fname) if rel_dir else fname
                csv_paths.append(rel_file)

    if csv_paths:
        lines.append("\n" + "="*60)
        lines.append("TÓM TẮT CÁC FILE .CSV")
        lines.append("="*60)

        for rel_path in csv_paths:
            abs_path = os.path.join(root_dir, rel_path)
            try:
                df = pd.read_csv(abs_path, nrows=sample_rows)
            except Exception as e:
                lines.append(f"\n⚠️  Không đọc được '{rel_path}': {e}")
                continue

            lines.append(f"\nCấu trúc file {rel_path}:")
            lines.append("Các cột: " + ", ".join(df.columns.astype(str)))
            lines.append("Một số dòng đầu tiên:")
            lines.append(df.to_string(index=False))

    summary_str = "\n".join(lines)
    return description, summary_str



DATA_SOURCE = ''
description, data_description = describe_dataset(DATA_SOURCE)
MODEL_POOL = [
    "https://huggingface.co/thanhtlx/image_classification_01",
    "https://huggingface.co/zhaospei/Model_3",
    "https://huggingface.co/zhaospei/Model_2", 
    "https://huggingface.co/zhaospei/Model_4",
    "https://huggingface.co/zhaospei/Model_7",
    "https://huggingface.co/zhaospei/Model_6",
    "https://huggingface.co/thanhtlx/text_classification_1",
]
    
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
            data_source=DATA_SOURCE,
            data_description=data_description
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