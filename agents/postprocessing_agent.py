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
        You are an AI Postprocessing Agent responsible for generating complete postprocessing code for any Machine Learning task.

        PROBLEM GUIDELINE:  
        {guideline}

        PROCESSING CODE:  
        {processing_code}

        ---

        🎯 Your task is to generate production-ready Python code for the postprocessing phase, tailored to the type of ML problem and the model’s outputs.

        You must return a **valid JSON object with exactly 4 fields**, corresponding to the following components:

        ---

        1. **"output_format"**:  
        - Describe the expected output format based on the task type (e.g. classification, regression, generation, object detection).  
        - Specify if the output should be in `.csv`, `.json`, `.txt`, etc.  
        - Mention any additional formatting (confidence scores, class labels, bounding boxes, etc.).

        2. **"result_aggregation"**:  
        - Python code that collects all model predictions from the processing stage.  
        - May include:
            - Merging batch outputs
            - Applying postprocessing logic (e.g. mapping class index to label)
            - Generating summaries/statistics (e.g. average score, class counts)

        3. **"export_code"**:  
        - Python code that saves the final results to file.  
        - Should support output to `.csv`, `.json`, or other formats as required.  
        - Should be scalable and support large datasets (use chunking if needed).  
        - May include saving logs, metrics, or additional artifacts.

        4. **"final_script"**:  
        - A fully integrated script that combines preprocessing, model inference, and postprocessing.  
        - Should include:
            - Argument parsing (for input/output paths)
            - Function calls to run preprocessing, inference, and postprocessing in sequence
            - Error handling and logging
            - Final export of results

        ---

        ⚠️ IMPORTANT:
        - The response **must be valid JSON** with exactly 4 fields:  
        `"output_format"`, `"result_aggregation"`, `"export_code"`, `"final_script"`  
        - Each value should be a **Python code string**, properly escaped and ready to execute.
        - Do not include markdown formatting, extra commentary, or any non-JSON content.
        - Ensure that the generated code matches the task type detected from the guideline.


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
            
            print("✅ Postprocessing: Hoàn thành output processing code!")
            return code
            
        except Exception as e:
            print(f"❌ Postprocessing: Lỗi trong quá trình tạo postprocessing code: {e}")
            print(f"🐛 Postprocessing: Error type: {type(e).__name__}")
            raise
