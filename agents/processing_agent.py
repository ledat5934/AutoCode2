from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import sys
import os
import time
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.data_models import Guideline, ModelSelection, PreprocessingCode, ProcessingCode

load_dotenv()

class ProcessingAgent:
    def __init__(self):
        print("⚙️ Processing Agent: Đang khởi tạo...")
        try:
            self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)
            print("✅ Processing Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Processing Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.parser = PydanticOutputParser(pydantic_object=ProcessingCode)
        print("✅ Processing Agent: Parser khởi tạo thành công")
        
        self.processing_template = ChatPromptTemplate.from_template("""
        You are an AI Processing Agent responsible for generating production-ready inference code for any Machine Learning task.

        PROBLEM GUIDELINE:  
        {guideline}

        SELECTED MODEL:  
        {selected_models}

        PREPROCESSING CODE:  
        {preprocessing_code}

        ---

        🎯 Your task: Create a complete and adaptable model inference code based on the problem and selected model. You must generate a valid JSON object with **exactly 3 fields**, each of which must be a properly escaped string containing valid Python code.

        ---

        1. **"model_loading"**:  
        - Load the selected model **directly from HuggingFace URL**  
        - Use `AutoModel` or `AutoModelFor...` as appropriate  
        - Do **NOT** load any submodels separately  
        - Load tokenizer/feature extractor if needed  
        - Optimize for device: use CPU/GPU/MPS with `torch.device` or similar setup  

        2. **"inference_pipeline"**:  
        - Build an inference pipeline based on the model type:  
            - **Computer Vision**: batch image inference, resize, normalize  
            - **NLP**: tokenization, attention masks, generation/classification  
            - **Audio**: load waveform, extract features, pass to model  
            - **Tabular**: load features, predict classification or regression  
        - Must be compatible with outputs from preprocessing  
        - Include logic for postprocessing if required (e.g., decoding, class mapping)

        3. **"main_processing_code"**:  
        - Integrate preprocessing output with inference pipeline  
        - Load data, initialize pipeline, run predictions  
        - Handle batch size, device placement, output formatting  
        - Log progress and errors  
        - Save or return results as appropriate for downstream use

        ---

        ⚠️ IMPORTANT:
        - Output MUST be a **valid JSON object with exactly 3 fields**:  
        `"model_loading"`, `"inference_pipeline"`, `"main_processing_code"`
        - Each value must be a Python code string, properly escaped and formatted
        - Handle multiple model types and ensure device compatibility
        - Do NOT include extra commentary, markdown formatting, or non-JSON output


        
        {format_instructions}
        """)
        print("✅ Processing Agent: Template khởi tạo thành công")

    def generate_processing_code(self, guideline: Guideline, selection: ModelSelection, preprocessing_code: PreprocessingCode) -> ProcessingCode:
        """Tạo code cho model inference cho bất kỳ ML task nào"""
        print("⚙️ Processing: Đang tạo adaptive model inference code...")
        print(f"📊 Input model: {len(selection.selected_models)} model")
        
        # Detect data type from guideline
        problem_text = guideline.problem_analysis.lower()
        if any(word in problem_text for word in ['image', 'picture', 'photo', 'visual', 'cv', 'computer vision']):
            data_type = "🖼️ Computer Vision"
        elif any(word in problem_text for word in ['text', 'nlp', 'language', 'sentiment', 'classification']):
            data_type = "📝 NLP"
        elif any(word in problem_text for word in ['audio', 'speech', 'sound', 'music']):
            data_type = "🔊 Audio"
        elif any(word in problem_text for word in ['time series', 'forecast', 'temporal', 'sequential']):
            data_type = "📈 Time Series"
        else:
            data_type = "📊 Tabular/Other"
        
        print(f"🔍 Detected data type: {data_type}")
        
        for model in selection.selected_models:
            model_name = model.split('/')[-1] if '/' in model else model
            print(f"   🤖 {model_name} for {data_type}")
        
        try:
            print("🔧 Processing: Đang tạo prompt...")
            prompt = self.processing_template.format_prompt(
                guideline=guideline.model_dump_json(indent=2),
                selected_models=selection.model_dump_json(indent=2),
                preprocessing_code=preprocessing_code.model_dump_json(indent=2),
                format_instructions=self.parser.get_format_instructions()
            )
            print("✅ Processing: Prompt tạo thành công")
            
            print("🌐 Processing: Đang gọi OpenAI API...")
            start_time = time.time()
            
            response = self.llm.invoke(prompt.to_messages())
            
            api_time = time.time() - start_time
            print(f"✅ Processing: API call thành công! Thời gian: {api_time:.2f}s")
            print(f"📄 Processing: Response length: {len(response.content)} characters")
            
            # Save raw response for debugging
            print("💾 Processing: Saving raw response for debugging...")
            with open("temp/raw_processing_response.txt", "w", encoding="utf-8") as f:
                f.write(response.content)
            
            print("🔍 Processing: Đang clean và parse response...")
            
            # Try to parse directly first
            try:
                code = self.parser.parse(response.content)
                print("✅ Processing: Direct parse thành công!")
            except Exception as parse_error:
                print(f"⚠️ Processing: Direct parse failed: {parse_error}")
                raise
            
            
            print("✅ Processing: Hoàn thành model inference code generation!")
            return code
            
        except Exception as e:
            print(f"❌ Processing: Critical error in processing code generation: {e}")
            print(f"🐛 Processing: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise