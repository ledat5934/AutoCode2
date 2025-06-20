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

from models.data_models import Guideline, ModelSelection, PreprocessingCode

load_dotenv()

class PreprocessingAgent:
    def __init__(self):
        print("🛠️ Preprocessing Agent: Đang khởi tạo...")
        try:
            self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)
            print("✅ Preprocessing Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Preprocessing Agent: Lỗi khởi tạo LLM: {e}")
            raise

        self.parser = PydanticOutputParser(pydantic_object=PreprocessingCode)
        print("✅ Preprocessing Agent: Parser khởi tạo thành công")

        self.preprocessing_template = ChatPromptTemplate.from_template("""
        You are an AI Preprocessing Agent specialized in generating preprocessing code for all types of Machine Learning problems.

        PROBLEM GUIDELINE: {guideline}

        SELECTED MODEL: {selected_models}

        DATA SOURCE: {data_source} contains input data files for preprocessing validation.
        Folder format description: {data_description}: it provide information about the folder structure and the files inside the folder.

        🎯 Task: Generate a complete, deployment-ready preprocessing code, compatible with all data types (Images / Text / Audio / Tabular / Time Series), with the following exact 4 fields in JSON format:

        "preprocessing_steps":

        Describe the sequential preprocessing steps appropriate for the data type.

        Example: loading data, missing value handling, normalization, encoding, augmentation, etc.

        The list must align with the specific data from the guideline.

        "code_imports":

        List of required libraries, grouped by data type:

        Images: torch, torchvision, PIL, cv2, albumentations

        Text: transformers, tokenizers, nltk, datasets

        Audio: librosa, torchaudio, soundfile

        Tabular/Time Series: pandas, numpy, sklearn, matplotlib, seaborn

        Always include: os, pathlib, json, logging

        "function":

        A list of individual preprocessing functions.

        Each function handles a specific task (data loading, outlier handling, masking, augmentation, etc.)

        Function names should be meaningful and reusable.

        "main_function":

        A master function that executes the entire preprocessing pipeline.

        Calls the above functions

        Automatically detects data type

        Splits data into train/validation

        Normalizes inputs according to the selected model

        Includes error handling and logging

        📌 IMPORTANT:

        The output MUST be a valid JSON object with exactly 4 fields as described.

        The code must be flexible and capable of handling multiple data types.

        Support for loading data from local path if necessary.

        Do not include any text outside the JSON.

        {format_instructions}
        """)
        print("✅ Preprocessing Agent: Template khởi tạo thành công")

    def generate_preprocessing_code(self, guideline: Guideline, selection: ModelSelection, data_source: str = "") -> PreprocessingCode:
        """Tạo code preprocessing cho bất kỳ loại data nào"""
        print("🛠️ Preprocessing: Đang tạo adaptive preprocessing code...")
        print(f"📊 Selected model: {len(selection.selected_models)} model")
        print(f"🔗 Data source: {data_source[:100]}..." if data_source else "🔗 No data source provided")
        
        for model in selection.selected_models:
            model_name = model.split('/')[-1] if '/' in model else model
            print(f"   🤖 {model_name}")
        
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
        
        try:
            print("🔧 Preprocessing: Đang tạo prompt...")
            prompt = self.preprocessing_template.format_prompt(
                guideline=guideline.model_dump_json(indent=2),
                selected_models=selection.model_dump_json(indent=2),
                data_source=data_source,
                format_instructions=self.parser.get_format_instructions()
            )
            print("✅ Preprocessing: Prompt tạo thành công")
            
            print("🌐 Preprocessing: Đang gọi OpenAI API...")
            start_time = time.time()
            
            response = self.llm.invoke(prompt.to_messages())
            
            api_time = time.time() - start_time
            print(f"✅ Preprocessing: API call thành công! Thời gian: {api_time:.2f}s")
            
            if response and response.content:
                print(f"📄 Preprocessing: Response length: {len(response.content)} characters")
                
                # Save raw response for debugging
                print("💾 Preprocessing: Saving raw response for debugging...")
                os.makedirs("temp", exist_ok=True)
                with open("temp/raw_preprocessing_response.txt", "w", encoding="utf-8") as f:
                    f.write(response.content)
                
                code = self.parser.parse(response.content)
                print("✅ Preprocessing: Parse thành công!")
                
                print("✅ Preprocessing: Hoàn thành code generation!")
                return code
            else:
                print("❌ Preprocessing: Response is None or empty!")
                raise Exception("Empty response from OpenAI API")
                
        except Exception as e:
            print(f"❌ Preprocessing: Lỗi trong quá trình tạo preprocessing code: {e}")
            print(f"🐛 Preprocessing: Error type: {type(e).__name__}")
            raise