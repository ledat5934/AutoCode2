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
from agents.validator_agent import ValidatorAgent

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
        self.validator = ValidatorAgent()
        
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

    def execute_pipeline(self, problem_description: str, model_pool: list, data_source: str = "", data_description: str = "") -> PipelineResult:
        """
        Execute the complete ML pipeline for ANY problem domain
        
        Args:
            problem_description (str): Description of the ML problem to solve
            model_pool (list): List of HuggingFace model URLs to choose from
            data_source (str): URL or path to data source (Google Drive, local path, etc.)
            data_description (str): Description of the data source
        Returns:
            PipelineResult: Complete execution result with generated solution
        """
        start_time = time.time()
        
        print("\n🎯 GENERAL ML PIPELINE - UNIVERSAL SOLUTION GENERATOR")
        print("="*80)
        print(f"📝 Problem: {problem_description}")
        print(f"🤖 Model Pool ({len(model_pool)} models):")
        print(f"🔗 Data Source: {data_source[:100]}..." if data_source else "🔗 No data source provided")
        print(f"🔗 Data Description: {data_description}")
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
            
            guideline = self.controller.create_guideline(problem_description, data_source, data_description)
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
                guideline, model_selection, data_source, data_description
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
            
            # STEP 11: Create complete solution script
            print("\n🔨 STEP 11: CREATING COMPLETE ML SOLUTION SCRIPT...")
            print("-" * 50)
            final_script_path = create_final_solution_script(self.results, OUTPUTS_DIR)
            print(f"✅ Generated solution script: {final_script_path}")
            
            # STEP 12: VALIDATION & EXECUTION
            print("\n🔍 STEP 12: CODE VALIDATION & EXECUTION")
            print("-" * 50)
            print("🛡️ Validator Agent checking and fixing code...")
            
            validation_result = self.validator.process_pipeline_result(final_script_path, data_source, data_description)
            
            # Debug validation result
            self.debug_print_result("VALIDATION RESULT", validation_result)
            
            execution_time = time.time() - start_time
            
            if validation_result.success:
                # Success với validator
                print("\n🎉 PIPELINE WITH VALIDATION COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"✅ Controller: Analyzed problem and designed solution approach")
                print(f"✅ Selector: Selected optimal model from {len(model_pool)} options")
                print(f"✅ Preprocessor: Generated adaptive data processing pipeline")
                print(f"✅ Processor: Created universal inference code")
                print(f"✅ Postprocessor: Generated appropriate output formatting")
                print(f"✅ Validator: Fixed code and executed successfully")
                print(f"📄 Generated CSV output: {validation_result.output_file}")
                print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
                
                # Display validation statistics
                if validation_result.statistics:
                    print(f"\n📊 OUTPUT STATISTICS:")
                    stats = validation_result.statistics
                    if 'total_rows' in stats:
                        print(f"   📝 Total rows: {stats['total_rows']}")
                    if 'columns' in stats:
                        print(f"   📋 Columns: {stats['columns']}")
                    if 'class_distribution' in stats:
                        print(f"   📈 Class distribution: {stats['class_distribution']}")
                    if 'format_valid' in stats:
                        print(f"   ✅ Format validation: {stats['format_valid']}")
                
                # Return successful result with validation info
                result = PipelineResult(
                    success=True,
                    output_file=validation_result.output_file,
                    execution_time=execution_time,
                    statistics={
                        "selected_models": len(model_selection.selected_models),
                        "processing_time": execution_time,
                        "validation_time": validation_result.execution_time,
                        "validation_stats": validation_result.statistics,
                        "timestamp": datetime.now().isoformat(),
                        "data_source": data_source,
                        "problem_type": problem_description[:100],
                        "final_script": final_script_path,
                        "csv_output": validation_result.output_file
                    },
                    error_message=None
                )
                
            else:
                # Validation failed
                print("\n⚠️ PIPELINE COMPLETED BUT VALIDATION FAILED")
                print("="*60)
                print(f"✅ Code generation completed successfully")
                print(f"❌ Code validation/execution failed")
                print(f"💾 Generated script available: {final_script_path}")
                print(f"⚠️ Validation error: {validation_result.error_message}")
                print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
                
                # Return partial success result
                result = PipelineResult(
                    success=False,
                    output_file=final_script_path,
                    execution_time=execution_time,
                    statistics={
                        "selected_models": len(model_selection.selected_models),
                        "processing_time": execution_time,
                        "validation_error": validation_result.error_message,
                        "timestamp": datetime.now().isoformat(),
                        "data_source": data_source,
                        "problem_type": problem_description[:100],
                        "final_script": final_script_path
                    },
                    error_message=f"Validation failed: {validation_result.error_message}"
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n❌ PIPELINE EXECUTION FAILED!")
            print(f"⚠️ Error: {str(e)}")
            print(f"⏱️ Execution time before failure: {execution_time:.2f} seconds")
            
            return PipelineResult(
                success=False,
                output_file="",
                execution_time=execution_time,
                statistics={
                    "error_stage": "pipeline_execution",
                    "timestamp": datetime.now().isoformat()
                },
                error_message=str(e)
            )

    def execute_generated_script(self, script_path: str, data_folder: str = None):
        """
        DEPRECATED: Execute generated script manually
        
        This method is now replaced by the validator agent which
        automatically validates and executes the code in STEP 12
        """
        print("⚠️ This method is deprecated. Validation and execution")
        print("   is now handled automatically by the Validator Agent in STEP 12")
        print(f"📄 Script location: {script_path}")
        
        # Có thể giữ lại cho manual execution nếu cần
        if os.path.exists(script_path):
            try:
                print(f"🚀 Manually executing: {script_path}")
                result = subprocess.run([sys.executable, script_path], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ Manual execution successful!")
                    print(f"📋 Output: {result.stdout}")
                else:
                    print("❌ Manual execution failed!")
                    print(f"⚠️ Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("⏰ Manual execution timeout!")
            except Exception as e:
                print(f"❌ Manual execution error: {e}")
        else:
            print(f"❌ Script not found: {script_path}")

def main():
    """Main entry point with enhanced validation"""
    pipeline = GeneralMLPipeline()
    
    print("\n🎯 TESTING GENERAL ML PIPELINE WITH VALIDATION")
    print("=" * 60)
    
    # Test problem - credit card fraud detection  
    test_problem = """
    Credit Card Fraud Detection:
    
    Analyze credit card transaction data to detect fraudulent transactions.
    The dataset contains transaction features like Time, Amount, and anonymized features V1-V28.
    Build a binary classification model to predict fraud (1) vs normal (0) transactions.
    
    Requirements:
    - Input: CSV file with transaction data
    - Output: CSV file with ID and class (0 or 1) predictions
    - Handle class imbalance (fraud is rare)
    - Optimize for high recall on fraud detection
    """
    
    # Execute complete pipeline with validation
    result = pipeline.execute_pipeline(
        problem_description=test_problem,
        model_pool=DEFAULT_MODEL_POOL,
        data_source="Credit card transaction dataset for fraud detection testing",
        data_description=''
    )
    
    # Final summary
    print(f"\n🏁 FINAL PIPELINE RESULT:")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Output: {result.output_file}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Error: {result.error_message or 'None'}")
    print("=" * 60)
    
    if result.success:
        print(f"🎉 Pipeline completed successfully!")
        print(f"📄 CSV output ready: {result.output_file}")
        
        # Hiển thị sample output nếu có
        if result.statistics and 'validation_stats' in result.statistics:
            val_stats = result.statistics['validation_stats']
            if 'sample_data' in val_stats:
                print(f"\n📊 Sample output data:")
                print(val_stats['sample_data'])
    else:
        print(f"⚠️ Pipeline completed with issues")
        print(f"💡 Check logs and generated files for debugging")

if __name__ == "__main__":
    main()