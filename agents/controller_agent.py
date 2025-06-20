from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import sys
import os
import time
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.data_models import Guideline

load_dotenv()

class ControllerAgent:
    def __init__(self):
        print("🎯 Controller Agent: Đang khởi tạo...")
        try:
            # STRUCTURED OUTPUT CONFIGURATION
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            print("✅ Controller Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Controller Agent: Lỗi khởi tạo LLM: {e}")
            raise

        # Output parser
        self.guideline_parser = PydanticOutputParser(pydantic_object=Guideline)
        print("✅ Controller Agent: Parser khởi tạo thành công")
        
        self.guideline_template = ChatPromptTemplate.from_template("""
        You are an AI Controller Agent specialized in analyzing and planning for Machine Learning problems.
        Given:
            PROBLEM DESCRIPTION: {problem_description}
            DATA SOURCE: {data_source} link to the folder contains files used for testing the solution.
            Folder format description: {data_description}: it provide information about the folder structure and the files inside the folder.
        Generate a detailed guideline with EXACTLY 4 FIELDS as follows:
            "problem_analysis":
            - Identify the problem type (Classification / Regression / Detection / NLP / etc.)
            - Analyze input/output requirements
            - Determine the domain (Computer Vision, NLP, Time Series, Tabular Data, etc.)
            - Identify key challenges and constraints
            "solution_approach":
            - Recommend the overall pipeline architecture
            - Define model selection strategy
            - Define preprocessing strategy
            - Define postprocessing strategy
            - Summarize the proposed approach to solve the problem
            "data_requirements":
            - Analyze expected data format from the data source
            - Define preprocessing needs
            - Specify required transformations
            - Outline data validation requirements
            "pipeline_overview":
            - Provide a detailed step-by-step workflow
            - Describe the data flow architecture
            - Explain the model integration strategy
            - Specify output format
            - Include error handling approach
        IMPORTANT: The response MUST be valid JSON and include EXACTLY these 4 fields!
        
        {format_instructions}
        """)
        print("✅ Controller Agent: Template khởi tạo thành công")

    def create_guideline(self, problem_description: str, data_source: str = "") -> Guideline:
        """Tạo guideline chi tiết từ problem description và data source"""
        print("🎯 Controller: Đang phân tích problem và tạo guideline...")
        print(f"📝 Problem description length: {len(problem_description)} characters")
        print(f"🔗 Data source: {data_source[:100]}..." if data_source else "🔗 No data source provided")
        
        try:
            print("🔧 Controller: Đang tạo prompt...")
            prompt = self.guideline_template.format_prompt(
                problem_description=problem_description,
                data_source=data_source,
                format_instructions=self.guideline_parser.get_format_instructions()
            )
            print("✅ Controller: Prompt tạo thành công")
            
            print("🌐 Controller: Đang gọi Gemini API...")
            start_time = time.time()
            
            response = self.llm.invoke(prompt.to_messages())
            
            api_time = time.time() - start_time
            print(f"✅ Controller: API call thành công! Thời gian: {api_time:.2f}s")
            print(f"📄 Controller: Response length: {len(response.content)} characters")
            guideline = self.guideline_parser.parse(response.content)
            print("✅ Controller: Parse thành công")
            # Debug output
            print(f"📊 Controller Result Preview:")
            print(f"   🔍 Problem Analysis: {guideline.problem_analysis[:100]}...")
            print(f"   🎯 Solution Approach: {guideline.solution_approach[:100]}...")
            print(f"   📊 Data Requirements: {guideline.data_requirements[:100]}...")
            
            print("✅ Controller: Hoàn thành guideline creation!")
            return guideline
        
        except Exception as e:
            print(f"❌ Controller: Lỗi trong quá trình tạo guideline: {e}")
            print(f"🐛 Controller: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise