from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import sys
import os
import time
import requests
from bs4 import BeautifulSoup
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.data_models import Guideline, ModelSelection

load_dotenv()

class SelectorAgent:
    def __init__(self):
        print("🤖 Selector Agent: Đang khởi tạo...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            print("✅ Selector Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Selector Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.parser = PydanticOutputParser(pydantic_object=ModelSelection)
        print("✅ Selector Agent: Parser khởi tạo thành công")
        
        # Template phân tích problem từ guideline
        self.problem_analysis_template = ChatPromptTemplate.from_template("""
        Bạn là AI expert phân tích bài toán ML từ guideline.
        
        GUIDELINE: {guideline}
        
        Hãy phân tích và xác định:
        1. Task type (Classification/Regression/Generation/Detection/Segmentation/etc.)
        2. Domain (Computer Vision/NLP/Audio/Tabular/Multimodal/etc.)
        3. Input data type (Images/Text/Audio/Numerical/etc.)
        4. Output format (Labels/Probabilities/Bounding boxes/Text/etc.)
        5. Specific requirements và constraints
        
        Trả về analysis dưới dạng structured format.
        """)
        
        # Template phân tích model từ HuggingFace content
        self.model_analysis_template = ChatPromptTemplate.from_template("""
        Bạn là AI expert phân tích models từ HuggingFace.
        
        MODEL URL: {model_url}
        MODEL CONTENT: {model_content}
        
        Từ nội dung trang HuggingFace, hãy phân tích:
        1. Model name và architecture type
        2. Task type model được train cho (classification/detection/generation/etc.)
        3. Domain và dataset được pretrain (ImageNet/COCO/BERT/etc.)
        4. Input format model chấp nhận (image size/text format/etc.)
        5. Output format model sinh ra (logits/embeddings/bboxes/etc.)
        6. Model performance và capabilities
        
        Trả về analysis chi tiết về model capabilities.
        """)
        
        # Template đánh giá compatibility và selection
        self.selection_template = ChatPromptTemplate.from_template("""
        Bạn là AI Model Selector chuyên chọn model tốt nhất cho bài toán ML.
        
        PROBLEM ANALYSIS: {problem_analysis}
        
        MODELS ANALYZED:
        {models_analysis}
        
        Nhiệm vụ:
        1. 🎯 COMPATIBILITY ASSESSMENT:
           - So sánh task types: problem vs models
           - Đánh giá domain alignment
           - Kiểm tra input/output compatibility
           - Xem xét transfer learning potential
        
        2. 🏆 MODEL SELECTION:
           - Chọn CHÍNH XÁC 1 model tốt nhất phù hợp nhất
           - Ưu tiên: Task relevance > Domain alignment > Performance
           - Consider ease of adaptation nếu cần
        
        3. 📋 SELECTION REASONING:
           - Explain tại sao model này là best choice
           - Identify required adaptations/fine-tuning
           - Mention pros/cons compared to other options
        
        4. 📊 OUTPUT FORMAT:
           - selected_models: [1 best model URL]
           - model_purposes: {{model_url: purpose_description}}
           - model_descriptions: {{model_url: detailed_description}}
           - model_inputs: {{model_url: input_format}}
           - model_outputs: {{model_url: output_format}}
           - selection_reasoning: detailed explanation
        
        Chọn model dựa trên analysis thực tế từ HuggingFace content!
        
        {format_instructions}
        """)
        
        print("✅ Selector Agent: Templates khởi tạo thành công")

    def fetch_model_content(self, model_url: str) -> str:
        """Truy cập và lấy nội dung từ HuggingFace model URL"""
        print(f"🔍 Selector: Đang truy cập {model_url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(model_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content_parts = []
            
            # Title
            title = soup.find('title')
            if title:
                content_parts.append(f"Title: {title.get_text().strip()}")
            
            # Model card/README content
            readme_sections = soup.find_all(['div', 'section'], 
                class_=lambda x: x and any(keyword in x.lower() for keyword in ['readme', 'model', 'description', 'card']))
            
            for section in readme_sections[:3]:
                text = section.get_text().strip()
                if text and len(text) > 50:
                    content_parts.append(text[:1500])  # Limit length
            
            # Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                content_parts.append(f"Meta: {meta_desc.get('content', '')}")
            
            # Tags/labels
            tags = soup.find_all(['span', 'div'], 
                class_=lambda x: x and any(keyword in x.lower() for keyword in ['tag', 'label', 'badge']))
            
            if tags:
                tag_texts = [tag.get_text().strip() for tag in tags[:10]]
                content_parts.append(f"Tags: {', '.join(tag_texts)}")
            
            combined_content = '\n\n'.join(content_parts)
            
            if not combined_content.strip():
                return f"Không thể lấy nội dung từ {model_url}. Chỉ có URL để analysis."
            
            print(f"✅ Selector: Đã lấy {len(combined_content)} chars content")
            return combined_content[:4000]  # Limit for API
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Selector: Lỗi request {model_url}: {e}")
            return f"Lỗi truy cập {model_url}. Sẽ phân tích từ URL pattern."
        except Exception as e:
            print(f"❌ Selector: Lỗi parse content {model_url}: {e}")
            return f"Lỗi xử lý content từ {model_url}."

    def analyze_problem_from_guideline(self, guideline: Guideline) -> str:
        """LLM phân tích problem từ guideline"""
        print("🧠 Selector: LLM đang phân tích problem từ guideline...")
        
        prompt = self.problem_analysis_template.format_prompt(
            guideline=guideline.model_dump_json(indent=2)
        )
        
        response = self.llm.invoke(prompt.to_messages())
        problem_analysis = response.content
        
        print("✅ Selector: Hoàn thành problem analysis")
        return problem_analysis

    def analyze_models_from_urls(self, model_pool: list) -> str:
        """LLM phân tích tất cả models từ URLs"""
        print(f"🔬 Selector: LLM đang phân tích {len(model_pool)} models từ URLs...")
        
        models_analysis = []
        
        for i, model_url in enumerate(model_pool, 1):
            print(f"📊 Selector: Analyzing model {i}/{len(model_pool)}: {model_url}")
            
            # Fetch model content
            model_content = self.fetch_model_content(model_url)
            
            # LLM analyze model
            prompt = self.model_analysis_template.format_prompt(
                model_url=model_url,
                model_content=model_content
            )
            
            try:
                response = self.llm.invoke(prompt.to_messages())
                analysis = f"Model {i}: {model_url}\n{response.content}\n" + "="*80
                models_analysis.append(analysis)
                time.sleep(1)  # Avoid rate limiting
                
            except Exception as e:
                print(f"❌ Selector: Lỗi analyze model {model_url}: {e}")
                models_analysis.append(f"Model {i}: {model_url}\nLỗi phân tích: {e}\n" + "="*80)
                continue
        
        combined_analysis = "\n\n".join(models_analysis)
        print("✅ Selector: Hoàn thành models analysis")
        return combined_analysis

    def select_models(self, guideline: Guideline, model_pool: list) -> ModelSelection:
        """Main method: Chọn model tốt nhất sử dụng URL crawling approach"""
        print("🤖 Selector: Bắt đầu intelligent model selection...")
        print(f"📊 Selector: Analyzing {len(model_pool)} models từ HuggingFace URLs")
        print("="*80)
        
        try:
            # Step 1: LLM analyze problem từ guideline
            problem_analysis = self.analyze_problem_from_guideline(guideline)
            
            # Step 2: LLM analyze all models từ URLs
            models_analysis = self.analyze_models_from_urls(model_pool)
            
            # Step 3: LLM chọn model tốt nhất
            print("🎯 Selector: LLM đang chọn model tốt nhất...")
            
            prompt = self.selection_template.format_prompt(
                problem_analysis=problem_analysis,
                models_analysis=models_analysis,
                format_instructions=self.parser.get_format_instructions()
            )
            
            response = self.llm.invoke(prompt.to_messages())
            selection = self.parser.parse(response.content)
            
            # Validation và logging
            if not selection.selected_models:
                print("⚠️ Selector: Không có model nào được chọn, sẽ fallback")
                selection.selected_models = [model_pool[0]]
                selection.selection_reasoning = "Fallback to first model due to selection failure"
            
            # Debug output
            print(f"🎯 Selector: FINAL SELECTION RESULTS:")
            print(f"   ✅ Best Model: {selection.selected_models[0] if selection.selected_models else 'None'}")
            print(f"   💭 Reasoning: {selection.selection_reasoning[:200]}...")
            print(f"   📊 Total analyzed: {len(model_pool)} models")
            
            print("✅ Selector: Hoàn thành URL-based model selection!")
            return selection
            
        except Exception as e:
            print(f"❌ Selector: Lỗi trong selection process: {e}")
            print(f"🐛 Selector: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Fallback selection
            print("🔄 Selector: Thực hiện fallback selection...")
            fallback_selection = ModelSelection(
                selected_models=[model_pool[0]] if model_pool else [],
                model_purposes={model_pool[0]: "Fallback selection"} if model_pool else {},
                model_descriptions={model_pool[0]: "Selected due to analysis failure"} if model_pool else {},
                model_inputs={model_pool[0]: "Unknown input format"} if model_pool else {},
                model_outputs={model_pool[0]: "Unknown output format"} if model_pool else {},
                selection_reasoning=f"Fallback selection due to error: {str(e)}"
            )
            return fallback_selection