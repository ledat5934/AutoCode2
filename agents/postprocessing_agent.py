from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.data_models import Guideline, ProcessingCode, PostprocessingCode

load_dotenv()

class PostprocessingAgent:
    def __init__(self):
        print("📊 Postprocessing Agent: Đang khởi tạo...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)
            print("✅ Postprocessing Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Postprocessing Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.parser = PydanticOutputParser(pydantic_object=PostprocessingCode)
        print("✅ Postprocessing Agent: Parser khởi tạo thành công")
        
        self.postprocessing_template = ChatPromptTemplate.from_template("""
        Bạn là AI Postprocessing Agent chuyên tạo output processing code cho mọi loại ML task.
        
        PROBLEM GUIDELINE: {guideline}
        PROCESSING CODE: {processing_code}
        
        Tạo comprehensive postprocessing code cho ML output:
        
        1. 🔍 OUTPUT TYPE DETECTION:
           - Auto-detect từ guideline: Classification/Regression/Generation/etc
           - Determine output format: CSV/JSON/txt/predictions
           - Identify post-processing requirements
           - Assess output validation needs
        
        2. 🔄 ADAPTIVE OUTPUT FORMATTING:
           FOR CLASSIFICATION:
           - Convert class indices to readable labels
           - Include confidence scores
           - Format as CSV: file_name, prediction, confidence
           FOR REGRESSION:
           - Apply denormalization nếu cần
           - Format numerical predictions
           - Include error bounds nếu có
           FOR NLP GENERATION:
           - Clean generated text
           - Apply post-processing filters
           - Format as structured output
           FOR OBJECT DETECTION:
           - Convert bounding boxes to readable format
           - Include detection scores và class names
           - Format as JSON với coordinates
        
        3. 📊 RESULT AGGREGATION:
           - Collect all predictions từ processing step
           - Apply business logic transformations
           - Aggregate statistics và metrics
           - Handle batch results efficiently
           - Create summary reports
        
        4. 💾 SMART EXPORT FUNCTIONALITY:
           - Generate appropriate file formats
           - Create visualization plots nếu cần
           - Export model performance metrics
           - Save intermediate results for debugging
           - Handle large datasets với chunking
        
        5. 🎯 FINAL SCRIPT INTEGRATION:
           - Combine tất cả preprocessing + processing + postprocessing
           - Create main execution function
           - Add argument parsing cho flexibility
           - Include comprehensive error handling
           - Generate complete runnable solution
        
        Generate complete solution producing the appropriate output format for the specific ML task.
        
        IMPORTANT: 
        - Output format MUST match task requirements
        - Ensure production-ready code quality
        
        {format_instructions}
        """)
        print("✅ Postprocessing Agent: Template khởi tạo thành công")

    def generate_postprocessing_code(self, guideline: Guideline, processing_code: ProcessingCode) -> PostprocessingCode:
        """Tạo code cho output processing và final script cho bất kỳ ML task nào"""
        print("📊 Postprocessing: Đang tạo adaptive output processing...")
        
        # Detect task type from guideline
        problem_text = guideline.problem_analysis.lower()
        if any(word in problem_text for word in ['classification', 'classify', 'predict class', 'categories']):
            task_type = "🏷️ Classification"
        elif any(word in problem_text for word in ['regression', 'predict value', 'forecast', 'numerical']):
            task_type = "📈 Regression"
        elif any(word in problem_text for word in ['generation', 'generate', 'synthesis', 'create']):
            task_type = "🎨 Generation"
        elif any(word in problem_text for word in ['detection', 'object', 'bounding box', 'localization']):
            task_type = "🎯 Detection"
        elif any(word in problem_text for word in ['segmentation', 'mask', 'pixel-wise']):
            task_type = "🖼️ Segmentation"
        else:
            task_type = "🔧 General"
        
        print(f"🔍 Detected task type: {task_type}")
        
        try:
            print("🔧 Postprocessing: Đang tạo prompt...")
            prompt = self.postprocessing_template.format_prompt(
                guideline=guideline.model_dump_json(indent=2),
                processing_code=processing_code.model_dump_json(indent=2),
                format_instructions=self.parser.get_format_instructions()
            )
            print("✅ Postprocessing: Prompt tạo thành công")
            
            print("🌐 Postprocessing: Đang gọi Gemini API...")
            start_time = time.time()
            
            response = self.llm.invoke(prompt.to_messages())
            
            api_time = time.time() - start_time
            print(f"✅ Postprocessing: API call thành công! Thời gian: {api_time:.2f}s")
            print(f"📄 Postprocessing: Response length: {len(response.content)} characters")
            
            print("🔍 Postprocessing: Đang parse response...")
            code = self.parser.parse(response.content)
            print("✅ Postprocessing: Parse thành công!")
            
            # Debug output
            print(f"📊 Postprocessing Result Preview:")
            print(f"   📝 Output Formatting: {len(code.output_formatting)} characters")
            print(f"   📊 Result Aggregation: {len(code.result_aggregation)} characters")
            print(f"   💾 Export Code: {len(code.export_code)} characters")
            print(f"   🎯 Final Script: {len(code.final_script)} characters")
            
            print("✅ Postprocessing: Hoàn thành output processing code!")
            return code
            
        except Exception as e:
            print(f"❌ Postprocessing: Lỗi trong quá trình tạo postprocessing code: {e}")
            print(f"🐛 Postprocessing: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Create adaptive fallback based on task type
            print(f"🔧 Postprocessing: Creating adaptive fallback for {task_type}...")
            
            if "Classification" in task_type:
                fallback_code = self._create_classification_fallback()
            elif "Regression" in task_type:
                fallback_code = self._create_regression_fallback()
            elif "Generation" in task_type:
                fallback_code = self._create_generation_fallback()
            elif "Detection" in task_type:
                fallback_code = self._create_detection_fallback()
            elif "Segmentation" in task_type:
                fallback_code = self._create_segmentation_fallback()
            else:
                fallback_code = self._create_general_fallback()
            
            print("✅ Postprocessing: Adaptive fallback created")
            return fallback_code

    def _create_classification_fallback(self):
        """Classification task fallback"""
        return PostprocessingCode(
            output_formatting="def format_classification_output(predictions, filenames, class_names=None):\n    results = []\n    for filename, pred in zip(filenames, predictions):\n        class_idx = pred if isinstance(pred, int) else int(pred)\n        class_name = class_names[class_idx] if class_names else f'class_{class_idx}'\n        results.append({'file_name': filename, 'prediction': class_name})\n    return results",
            result_aggregation="# Aggregate classification results\nall_predictions = []\nall_filenames = []\nfor filename, prediction in results.items():\n    all_filenames.append(filename)\n    all_predictions.append(prediction)\nformatted_results = format_classification_output(all_predictions, all_filenames)",
            visualization_code="import matplotlib.pyplot as plt\nimport seaborn as sns\nfrom collections import Counter\n\ndef create_prediction_histogram(predictions):\n    pred_counts = Counter([r['prediction'] for r in predictions])\n    plt.figure(figsize=(10, 6))\n    sns.barplot(x=list(pred_counts.keys()), y=list(pred_counts.values()))\n    plt.title('Prediction Distribution')\n    plt.xticks(rotation=45)\n    plt.tight_layout()\n    plt.savefig('prediction_distribution.png')\n    plt.close()",
            export_code="import pandas as pd\n\ndef export_classification_results(results, output_file='predictions.csv'):\n    df = pd.DataFrame(results)\n    df.to_csv(output_file, index=False)\n    print(f'Results exported to {output_file}')\n    return output_file",
            final_script="# Export final results\noutput_file = export_classification_results(formatted_results)\ncreate_prediction_histogram(formatted_results)\nprint(f'Classification completed! Results saved to {output_file}')"
        )

    def _create_regression_fallback(self):
        """Regression task fallback"""
        return PostprocessingCode(
            output_formatting="def format_regression_output(predictions, filenames):\n    results = []\n    for filename, pred in zip(filenames, predictions):\n        pred_value = float(pred) if not isinstance(pred, float) else pred\n        results.append({'file_name': filename, 'prediction': pred_value})\n    return results",
            result_aggregation="# Aggregate regression results\nall_predictions = []\nall_filenames = []\nfor filename, prediction in results.items():\n    all_filenames.append(filename)\n    all_predictions.append(prediction)\nformatted_results = format_regression_output(all_predictions, all_filenames)",
            visualization_code="import matplotlib.pyplot as plt\nimport numpy as np\n\ndef create_prediction_plot(predictions):\n    pred_values = [r['prediction'] for r in predictions]\n    plt.figure(figsize=(12, 6))\n    plt.subplot(1, 2, 1)\n    plt.hist(pred_values, bins=30, alpha=0.7)\n    plt.title('Prediction Distribution')\n    plt.xlabel('Predicted Values')\n    plt.ylabel('Frequency')\n    plt.subplot(1, 2, 2)\n    plt.plot(pred_values)\n    plt.title('Predictions Over Samples')\n    plt.xlabel('Sample Index')\n    plt.ylabel('Predicted Value')\n    plt.tight_layout()\n    plt.savefig('regression_results.png')\n    plt.close()",
            export_code="import pandas as pd\nimport numpy as np\n\ndef export_regression_results(results, output_file='predictions.csv'):\n    df = pd.DataFrame(results)\n    # Add statistics\n    pred_values = df['prediction'].values\n    stats = {\n        'mean': np.mean(pred_values),\n        'std': np.std(pred_values),\n        'min': np.min(pred_values),\n        'max': np.max(pred_values)\n    }\n    df.to_csv(output_file, index=False)\n    print(f'Results exported to {output_file}')\n    print(f'Statistics: {stats}')\n    return output_file",
            final_script="# Export final results\noutput_file = export_regression_results(formatted_results)\ncreate_prediction_plot(formatted_results)\nprint(f'Regression completed! Results saved to {output_file}')"
        )

    def _create_generation_fallback(self):
        """Generation task fallback"""
        return PostprocessingCode(
            output_formatting="def format_generation_output(generated_texts, prompts):\n    results = []\n    for prompt, generated in zip(prompts, generated_texts):\n        cleaned_text = generated.strip()\n        results.append({'prompt': prompt, 'generated_text': cleaned_text, 'length': len(cleaned_text)})\n    return results",
            result_aggregation="# Aggregate generation results\nall_generated = []\nall_prompts = []\nfor prompt, generated in results.items():\n    all_prompts.append(prompt)\n    all_generated.append(generated)\nformatted_results = format_generation_output(all_generated, all_prompts)",
            visualization_code="import matplotlib.pyplot as plt\n\ndef create_generation_stats(results):\n    lengths = [r['length'] for r in results]\n    plt.figure(figsize=(10, 6))\n    plt.hist(lengths, bins=20, alpha=0.7)\n    plt.title('Generated Text Length Distribution')\n    plt.xlabel('Text Length (characters)')\n    plt.ylabel('Frequency')\n    plt.savefig('generation_stats.png')\n    plt.close()",
            export_code="import json\nimport pandas as pd\n\ndef export_generation_results(results, output_file='generated_texts.json'):\n    with open(output_file, 'w', encoding='utf-8') as f:\n        json.dump(results, f, ensure_ascii=False, indent=2)\n    # Also create CSV for easy viewing\n    df = pd.DataFrame(results)\n    df.to_csv(output_file.replace('.json', '.csv'), index=False)\n    print(f'Results exported to {output_file}')\n    return output_file",
            final_script="# Export final results\noutput_file = export_generation_results(formatted_results)\ncreate_generation_stats(formatted_results)\nprint(f'Text generation completed! Results saved to {output_file}')"
        )

    def _create_detection_fallback(self):
        """Object detection task fallback"""
        return PostprocessingCode(
            output_formatting="def format_detection_output(detections, filenames):\n    results = []\n    for filename, dets in zip(filenames, detections):\n        file_detections = []\n        for det in dets:\n            file_detections.append({\n                'bbox': det['bbox'],\n                'class': det['class'],\n                'confidence': det['confidence']\n            })\n        results.append({'file_name': filename, 'detections': file_detections})\n    return results",
            result_aggregation="# Aggregate detection results\nall_detections = []\nall_filenames = []\nfor filename, detections in results.items():\n    all_filenames.append(filename)\n    all_detections.append(detections)\nformatted_results = format_detection_output(all_detections, all_filenames)",
            visualization_code="import matplotlib.pyplot as plt\nfrom collections import Counter\n\ndef create_detection_stats(results):\n    all_classes = []\n    detection_counts = []\n    for result in results:\n        det_count = len(result['detections'])\n        detection_counts.append(det_count)\n        for det in result['detections']:\n            all_classes.append(det['class'])\n    \n    plt.figure(figsize=(15, 5))\n    plt.subplot(1, 3, 1)\n    plt.hist(detection_counts, bins=20)\n    plt.title('Detections per Image')\n    plt.xlabel('Number of Detections')\n    plt.ylabel('Frequency')\n    \n    class_counts = Counter(all_classes)\n    plt.subplot(1, 3, 2)\n    plt.bar(class_counts.keys(), class_counts.values())\n    plt.title('Class Distribution')\n    plt.xticks(rotation=45)\n    \n    plt.tight_layout()\n    plt.savefig('detection_stats.png')\n    plt.close()",
            export_code="import json\n\ndef export_detection_results(results, output_file='detections.json'):\n    with open(output_file, 'w') as f:\n        json.dump(results, f, indent=2)\n    print(f'Detection results exported to {output_file}')\n    return output_file",
            final_script="# Export final results\noutput_file = export_detection_results(formatted_results)\ncreate_detection_stats(formatted_results)\nprint(f'Object detection completed! Results saved to {output_file}')"
        )

    def _create_segmentation_fallback(self):
        """Segmentation task fallback"""
        return PostprocessingCode(
            output_formatting="def format_segmentation_output(masks, filenames):\n    results = []\n    for filename, mask in zip(filenames, masks):\n        mask_stats = {\n            'shape': mask.shape,\n            'unique_classes': len(np.unique(mask)),\n            'mask_file': filename.replace('.jpg', '_mask.png').replace('.png', '_mask.png')\n        }\n        results.append({'file_name': filename, 'mask_stats': mask_stats})\n    return results",
            result_aggregation="# Aggregate segmentation results\nall_masks = []\nall_filenames = []\nfor filename, mask in results.items():\n    all_filenames.append(filename)\n    all_masks.append(mask)\nformatted_results = format_segmentation_output(all_masks, all_filenames)",
            visualization_code="import matplotlib.pyplot as plt\nimport numpy as np\n\ndef create_segmentation_stats(results):\n    class_counts = [r['mask_stats']['unique_classes'] for r in results]\n    plt.figure(figsize=(10, 6))\n    plt.hist(class_counts, bins=20)\n    plt.title('Number of Classes per Segmentation')\n    plt.xlabel('Number of Classes')\n    plt.ylabel('Frequency')\n    plt.savefig('segmentation_stats.png')\n    plt.close()",
            export_code="import json\nimport numpy as np\nfrom PIL import Image\n\ndef export_segmentation_results(results, output_file='segmentation_results.json'):\n    # Save mask statistics\n    with open(output_file, 'w') as f:\n        json.dump(results, f, indent=2)\n    print(f'Segmentation results exported to {output_file}')\n    return output_file",
            final_script="# Export final results\noutput_file = export_segmentation_results(formatted_results)\ncreate_segmentation_stats(formatted_results)\nprint(f'Segmentation completed! Results saved to {output_file}')"
        )

    def _create_general_fallback(self):
        """General task fallback"""
        return PostprocessingCode(
            output_formatting="def format_general_output(predictions, identifiers):\n    results = []\n    for identifier, pred in zip(identifiers, predictions):\n        results.append({'identifier': identifier, 'prediction': str(pred)})\n    return results",
            result_aggregation="# Aggregate general results\nall_predictions = []\nall_identifiers = []\nfor identifier, prediction in results.items():\n    all_identifiers.append(identifier)\n    all_predictions.append(prediction)\nformatted_results = format_general_output(all_predictions, all_identifiers)",
            visualization_code="import matplotlib.pyplot as plt\n\ndef create_general_stats(results):\n    print(f'Total predictions: {len(results)}')\n    # Basic visualization\n    plt.figure(figsize=(8, 6))\n    plt.bar(range(min(10, len(results))), [hash(str(r['prediction'])) % 100 for r in results[:10]])\n    plt.title('Sample Predictions')\n    plt.xlabel('Sample Index')\n    plt.ylabel('Prediction Hash')\n    plt.savefig('general_results.png')\n    plt.close()",
            export_code="import pandas as pd\nimport json\n\ndef export_general_results(results, output_file='predictions.csv'):\n    df = pd.DataFrame(results)\n    df.to_csv(output_file, index=False)\n    # Also save as JSON\n    with open(output_file.replace('.csv', '.json'), 'w') as f:\n        json.dump(results, f, indent=2)\n    print(f'Results exported to {output_file}')\n    return output_file",
            final_script="# Export final results\noutput_file = export_general_results(formatted_results)\ncreate_general_stats(formatted_results)\nprint(f'Processing completed! Results saved to {output_file}')"
        )