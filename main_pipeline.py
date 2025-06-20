#!/usr/bin/env python3
"""
GENERAL ML PIPELINE EXECUTOR
Orchestrates the entire multiagent system to generate ML solutions for ANY domain
"""

import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime
import time
import json

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

from agents.controller_agent import ControllerAgent
from agents.selector_agent import SelectorAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.processing_agent import ProcessingAgent
from agents.postprocessing_agent import PostprocessingAgent

from config import DEFAULT_PROBLEM, DEFAULT_MODEL_POOL, OUTPUTS_DIR, TEMP_DIR
from utils import save_intermediate_result, create_final_solution_script, display_pipeline_summary
from models.data_models import PipelineResult

class GeneralMLPipeline:
    """General ML Pipeline that can handle ANY machine learning task"""
    
    def __init__(self):
        """Initialize all agents"""
        print("🚀 Initializing General ML Pipeline...")
        
        self.controller = ControllerAgent()
        self.selector = SelectorAgent()
        self.preprocessor = PreprocessingAgent()
        self.processor = ProcessingAgent()
        self.postprocessor = PostprocessingAgent()
        
        # Storage for results
        self.results = {}
        
        print("✅ All agents initialized!")

    def debug_print_result(self, step_name: str, result: any, max_chars: int = 500):
        """Print debug info for each step"""
        print(f"\n🔍 DEBUG - {step_name}:")
        print("=" * 60)
        
        if hasattr(result, 'dict'):
            result_dict = result.dict()
            for key, value in result_dict.items():
                print(f"📌 {key.upper()}:")
                if isinstance(value, str):
                    if len(value) > max_chars:
                        print(f"   {value[:max_chars]}...")
                    else:
                        print(f"   {value}")
                elif isinstance(value, list):
                    print(f"   📋 List with {len(value)} items:")
                    for i, item in enumerate(value[:3]):  # Show first 3 items
                        print(f"      {i+1}. {str(item)[:100]}...")
                elif isinstance(value, dict):
                    print(f"   📚 Dict with {len(value)} keys:")
                    for k, v in list(value.items())[:3]:  # Show first 3 items
                        print(f"      {k}: {str(v)[:50]}...")
                else:
                    print(f"   {value}")
                print()
        else:
            print(f"Raw result: {str(result)[:max_chars]}...")
        
        print("=" * 60)

    def execute_pipeline(self, problem_description: str, model_pool: list, data_source: str = "") -> PipelineResult:
        """
        Execute the complete ML pipeline for ANY problem domain
        
        Args:
            problem_description (str): Description of the ML problem to solve
            model_pool (list): List of HuggingFace model URLs to choose from
            data_source (str): URL or path to data source (Google Drive, local path, etc.)
            
        Returns:
            PipelineResult: Complete execution result with generated solution
        """
        start_time = time.time()
        
        print("\n🎯 GENERAL ML PIPELINE - UNIVERSAL SOLUTION GENERATOR")
        print("="*80)
        print(f"📝 Problem: {problem_description}")
        print(f"🤖 Model Pool ({len(model_pool)} models):")
        for i, model in enumerate(model_pool, 1):
            model_name = model.split('/')[-1] if '/' in model else model
            print(f"   {i}. {model_name}")
        print(f"🔗 Data Source: {data_source[:100]}..." if data_source else "🔗 No data source provided")
        print("="*80)
        
        try:
            # STEP 1-2: Problem Analysis & Guideline Creation
            print("\n📋 STEP 1-2: PROBLEM ANALYSIS & SOLUTION DESIGN")
            print("-" * 50)
            print("🤖 Controller Agent analyzing problem and designing solution approach...")
            
            guideline = self.controller.create_guideline(problem_description, data_source)
            self.results['guideline'] = guideline
            save_intermediate_result('guideline', guideline, TEMP_DIR)
            
            self.debug_print_result("CONTROLLER ANALYSIS", guideline)
            
            # STEP 3-4: Intelligent Model Selection
            print("\n🤖 STEP 3-4: INTELLIGENT MODEL SELECTION")
            print("-" * 50)
            print("🎯 Selector Agent choosing optimal model for the task...")
            print(f"📋 Evaluating {len(model_pool)} available models")
            
            model_selection = self.selector.select_models(guideline, model_pool)
            self.results['model_selection'] = model_selection
            save_intermediate_result('model_selection', model_selection, TEMP_DIR)
            
            self.debug_print_result("MODEL SELECTION", model_selection)
            print(f"🎯 SELECTED MODEL: {model_selection.selected_models[0] if model_selection.selected_models else 'None'}")
            print(f"💭 REASONING: {model_selection.selection_reasoning}")
            
            # STEP 5-6: Adaptive Preprocessing Code Generation
            print("\n🛠️ STEP 5-6: ADAPTIVE DATA PREPROCESSING")
            print("-" * 50)
            print("⚙️ Preprocessing Agent generating data processing code...")
            
            preprocessing_code = self.preprocessor.generate_preprocessing_code(
                guideline, model_selection, data_source
            )
            self.results['preprocessing_code'] = preprocessing_code
            save_intermediate_result('preprocessing_code', preprocessing_code, TEMP_DIR)
            
            self.debug_print_result("PREPROCESSING CODE", preprocessing_code)
            
            # STEP 7-8: Universal Inference Code Generation
            print("\n⚙️ STEP 7-8: UNIVERSAL INFERENCE PIPELINE")
            print("-" * 50)
            print("🔄 Processing Agent generating model inference code...")
            
            processing_code = self.processor.generate_processing_code(
                guideline, model_selection, preprocessing_code
            )
            self.results['processing_code'] = processing_code
            save_intermediate_result('processing_code', processing_code, TEMP_DIR)
            
            self.debug_print_result("PROCESSING CODE", processing_code)
            
            # STEP 9-10: Adaptive Output Processing
            print("\n📊 STEP 9-10: ADAPTIVE OUTPUT PROCESSING")
            print("-" * 50)
            print("📈 Postprocessing Agent generating output formatting code...")
            
            postprocessing_code = self.postprocessor.generate_postprocessing_code(
                guideline, processing_code
            )
            self.results['postprocessing_code'] = postprocessing_code
            save_intermediate_result('postprocessing_code', postprocessing_code, TEMP_DIR)
            
            self.debug_print_result("POSTPROCESSING CODE", postprocessing_code)
            
            # Create complete solution script
            print("\n🔨 CREATING COMPLETE ML SOLUTION SCRIPT...")
            final_script_path = create_final_solution_script(self.results, OUTPUTS_DIR)
            
            execution_time = time.time() - start_time
            
            # Success summary
            print("\n🎉 PIPELINE EXECUTION SUMMARY:")
            print("="*60)
            print(f"✅ Controller: Analyzed problem and designed solution approach")
            print(f"✅ Selector: Selected optimal model from {len(model_pool)} options")
            print(f"✅ Preprocessor: Generated adaptive data processing pipeline")
            print(f"✅ Processor: Created universal inference code")
            print(f"✅ Postprocessor: Generated appropriate output formatting")
            print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
            print("="*60)
            
            # Create result object
            result = PipelineResult(
                success=True,
                output_file=final_script_path,
                execution_time=execution_time,
                statistics={
                    "selected_models": len(model_selection.selected_models),
                    "processing_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "data_source": data_source,
                    "problem_type": problem_description[:100]
                },
                error_message=None
            )
            
            print(f"\n🎉 GENERAL ML PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"📄 Complete solution script: {final_script_path}")
            print(f"💡 The generated script can handle your specific ML task!")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"\n❌ PIPELINE FAILED: {error_msg}")
            print(f"🐛 Debug info: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            result = PipelineResult(
                success=False,
                output_file="",
                execution_time=execution_time,
                statistics={},
                error_message=error_msg
            )
            
            return result

    def execute_generated_script(self, script_path: str, data_folder: str = None):
        """
        Execute the generated ML solution script
        
        Args:
            script_path (str): Path to the generated script
            data_folder (str): Optional path to data folder
        """
        print(f"\n🚀 EXECUTING GENERATED ML SOLUTION: {script_path}")
        print("="*60)
        
        # Check script exists
        script_file = Path(script_path)
        if not script_file.exists():
            print(f"❌ Script not found: {script_path}")
            return
        
        try:
            # Execute the script
            print("🔄 Running the generated ML solution...")
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for ML tasks
            )
            
            print("📤 SCRIPT OUTPUT:")
            print("-" * 40)
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                print("\n⚠️ WARNINGS/ERRORS:")
                print(result.stderr)
            
            print(f"\n✅ Script execution completed with return code: {result.returncode}")
            
            # Check for common output files
            output_files = []
            for pattern in ["*.csv", "*.json", "*.txt", "results.*", "output.*", "predictions.*"]:
                output_files.extend(Path.cwd().glob(pattern))
            
            if output_files:
                print(f"\n🎯 Generated output files:")
                for output_file in output_files:
                    print(f"   📄 {output_file.name} ({output_file.stat().st_size} bytes)")
                    
                    # Try to show sample content for small files
                    if output_file.suffix in ['.csv', '.json', '.txt'] and output_file.stat().st_size < 10000:
                        try:
                            print(f"   📋 Sample content from {output_file.name}:")
                            with open(output_file, 'r', encoding='utf-8') as f:
                                content = f.read(500)  # First 500 chars
                                print(f"      {content[:200]}{'...' if len(content) > 200 else ''}")
                        except Exception as e:
                            print(f"      ⚠️ Could not read content: {e}")
            else:
                print("\n⚠️ No output files found in current directory")
                
        except subprocess.TimeoutExpired:
            print("❌ Script execution timed out (10 minutes)")
        except Exception as e:
            print(f"❌ Error executing script: {e}")

def main():
    """Main entry point for general ML pipeline"""
    pipeline = GeneralMLPipeline()
    
    print("🎯 GENERAL ML PIPELINE - UNIVERSAL SOLUTION GENERATOR")
    print("="*70)
    print("🚀 This pipeline can solve ANY machine learning problem!")
    print("📋 Supported domains: Computer Vision, NLP, Audio, Tabular Data, Time Series")
    print("🤖 Works with any HuggingFace model")
    print("🔗 Supports various data sources (Google Drive, local files, URLs)")
    print("="*70)
    
    # Show example configurations
    
    print(f"\n💡 To use this pipeline, call:")
    print(f"   pipeline.execute_pipeline(problem_description, model_pool, data_source)")
    print(f"\n📄 Default configuration will be used for demonstration...")
    
    # Execute with default configuration
    result = pipeline.execute_pipeline(
        problem_description=DEFAULT_PROBLEM,
        model_pool=DEFAULT_MODEL_POOL,
        data_source=""
    )
    
    # Display summary
    display_pipeline_summary(pipeline.results)
    
    if result.success:
        print(f"\n📁 Intermediate files saved in: {TEMP_DIR}")
        print(f"📄 Final solution script in: {OUTPUTS_DIR}")
        print(f"\n🚀 To run the generated solution:")
        print(f"   python {result.output_file}")
        
        # List generated files
        temp_files = list(Path(TEMP_DIR).glob("*.json")) if Path(TEMP_DIR).exists() else []
        if temp_files:
            print(f"\n📋 Debug files available:")
            for f in temp_files:
                print(f"   📄 {f.name}")

if __name__ == "__main__":
    main()