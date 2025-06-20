from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.data_models import Guideline

load_dotenv()

class ControllerAgent:
    def __init__(self):
        print("🎯 Controller Agent: Đang khởi tạo...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            print("✅ Controller Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Controller Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.guideline_parser = PydanticOutputParser(pydantic_object=Guideline)
        print("✅ Controller Agent: Parser khởi tạo thành công")
        
        self.guideline_template = ChatPromptTemplate.from_template("""
        Bạn là AI Controller Agent chuyên phân tích và lập kế hoạch cho các bài toán Machine Learning.
        
        PROBLEM DESCRIPTION: {problem_description}
        DATA SOURCE: {data_source}
        
        Hãy tạo guideline chi tiết với CHÍNH XÁC 5 FIELDS sau:
        
        1. "problem_analysis": 
           - Xác định loại bài toán (Classification/Regression/Detection/NLP/etc)
           - Phân tích input/output requirements
           - Determine domain (Computer Vision, NLP, Time Series, Tabular Data)
           - Identify key challenges và constraints
        
        2. "solution_approach":
           - Recommend overall pipeline architecture
           - Define model selection strategy
           - Define preprocessing strategy  
           - Define postprocessing strategy
           - Approach để giải quyết problem
        
        3. "data_requirements":
           - Analyze expected data format từ data source
           - Define preprocessing needs
           - Specify required transformations
           - Data validation requirements
        
        4. "pipeline_overview":
           - Step-by-step workflow chi tiết
           - Data flow architecture
           - Model integration strategy
           - Output format specifications
           - Error handling approach
        
        5. "success_criteria":
           - Output quality requirements
           - Performance metrics
           - Deployment considerations
           - Validation criteria
        
        QUAN TRỌNG: Phải trả về JSON với CHÍNH XÁC 5 fields trên. Không được thiếu field nào!
        
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
            
            print("🔍 Controller: Đang clean và parse response...")
            
            # Clean response content
            cleaned_content = self._clean_json_content(response.content)
            
            guideline = self.guideline_parser.parse(cleaned_content)
            print("✅ Controller: Parse thành công!")
            
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

    def _clean_json_content(self, content: str) -> str:
        """Clean JSON content để fix escape và format issues"""
        try:
            # Extract JSON từ markdown code blocks nếu có
            if "```json" in content:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            # Fix missing comma issues trong JSON 
            import re
            import json
            
            # Fix missing comma between fields - tìm pattern }"field_name":
            content = re.sub(r'"\s*\n\s*"([^"]+)":', r'",\n  "\1":', content)
            
            # Fix newlines in string values
            content = re.sub(r'\\n', r'\\n', content)
            
            # Try to parse and reformat to ensure valid JSON
            try:
                # Test parse để check validity
                parsed = json.loads(content)
                # Reformat với proper spacing
                content = json.dumps(parsed, ensure_ascii=False, indent=2)
                print("✅ Controller: JSON successfully validated and reformatted")
            except json.JSONDecodeError as e:
                print(f"⚠️ Controller: JSON still invalid after cleaning: {e}")
                # Additional fixes for common issues
                content = content.replace('"\n  "', '",\n  "')
                content = content.replace('",\n}', '"\n}')
            
            # Save cleaned content for debugging
            import os
            os.makedirs("temp", exist_ok=True)
            with open("temp/cleaned_controller_response.txt", "w", encoding="utf-8") as f:
                f.write(content)
            
            print("✅ Controller: JSON content cleaned successfully")
            return content
            
        except Exception as e:
            print(f"⚠️ Controller: Error cleaning JSON content: {e}")
            return content