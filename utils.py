import os
import json
from pathlib import Path
from models.data_models import PreprocessingCode, ProcessingCode, PostprocessingCode, ModelSelection, Guideline

def save_intermediate_result(name: str, result, output_dir: str):
    """Lưu kết quả trung gian dưới dạng JSON"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.dict(), f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu {name} vào {filepath}")

def create_final_solution_script(results: dict, output_dir: str) -> str:
    """Kết hợp toàn bộ code thành 1 script hoàn chỉnh"""
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "final_solution.py")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated ML solution script\n")
        f.write("# =================================\n\n")
        
        # Gộp các đoạn mã
        f.write("# 📦 IMPORTS\n")
        f.write(results["preprocessing_code"].code_imports + "\n\n")
        
        f.write("# 🧹 PREPROCESSING FUNCTIONS\n")
        for func in results["preprocessing_code"].function:
            f.write(func + "\n\n")
        f.write("# 🚀 MAIN PREPROCESSING FUNCTION\n")
        f.write(results["preprocessing_code"].main_function + "\n\n")
        
        f.write("# 🤖 MODEL LOADING\n")
        f.write(results["processing_code"].model_loading + "\n\n")
        
        f.write("# 🔁 INFERENCE PIPELINE\n")
        f.write(results["processing_code"].inference_pipeline + "\n\n")
        
        f.write("# 🧠 MAIN PROCESSING CODE\n")
        f.write(results["processing_code"].main_processing_code + "\n\n")
        
        f.write("# 📊 RESULT AGGREGATION\n")
        f.write(results["postprocessing_code"].result_aggregation + "\n\n")
        
        f.write("# 💾 EXPORTING RESULTS\n")
        f.write(results["postprocessing_code"].export_code + "\n\n")
        
        f.write("# 🧩 FINAL INTEGRATED SCRIPT\n")
        f.write(results["postprocessing_code"].final_script + "\n\n")
        
    print(f"✅ Final solution script created at: {script_path}")
    return script_path

def display_pipeline_summary(results: dict):
    """Hiển thị tóm tắt thông tin các bước trong pipeline"""
    print("\n📊 PIPELINE SUMMARY:")
    print("=" * 60)
    print(f"📌 Problem Summary: {results['guideline'].problem_analysis[:80]}...")
    print(f"🧠 Selected Model(s): {', '.join(results['model_selection'].selected_models)}")
    print(f"🔧 Preprocessing: {len(results['preprocessing_code'].function)} functions generated")
    print(f"⚙️ Inference: {len(results['processing_code'].inference_pipeline)} chars of code")
    print(f"📈 Postprocessing Format: {results['postprocessing_code'].output_format[:80]}...")
    print("=" * 60)
