from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import sys
import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.data_models import PipelineResult

load_dotenv()

class ValidatorAgent:
    """
    Agent đơn giản: rewrite code bằng Gemini và execute để tạo CSV output
    """
    
    def __init__(self):
        print("🔍 Validator Agent: Đang khởi tạo...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            print("✅ Validator Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Validator Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Template để rewrite code
        self.rewrite_template = ChatPromptTemplate.from_template("""
        Bạn là expert Python developer. Nhiệm vụ: Rewrite code Python để chạy được và tạo file CSV output.

        CODE CẦN REWRITE:
        ```python
        {original_code}
        ```

        YÊU CẦU REWRITE:
        1. Sửa tất cả lỗi syntax, import, duplicate functions
        2. Tạo code chạy được ngay lập tức
        3. Output: file CSV 
        4. File output: 'predicted_results.csv'

        
        Chỉ trả về PYTHON CODE hoàn chỉnh, không cần explanation:
        """)

    def _setup_logging(self) -> logging.Logger:
        """Thiết lập logging đơn giản"""
        logger = logging.getLogger("ValidatorAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def rewrite_code_with_gemini(self, original_code: str) -> str:
        """
        Sử dụng Gemini để rewrite code hoàn toàn
        
        Args:
            original_code: Code gốc cần rewrite
            
        Returns:
            Code đã được rewrite
        """
        self.logger.info("🤖 Đang gọi Gemini để rewrite code...")
        
        try:
            prompt = self.rewrite_template.format_prompt(original_code=original_code)
            response = self.llm.invoke(prompt.to_messages())
            
            rewritten_code = response.content
            
            # Clean up markdown formatting
            if "```python" in rewritten_code:
                rewritten_code = rewritten_code.split("```python")[1].split("```")[0]
            elif "```" in rewritten_code:
                rewritten_code = rewritten_code.split("```")[1].split("```")[0]
                
            self.logger.info("✅ Gemini đã rewrite xong code")
            return rewritten_code.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi gọi Gemini: {e}")
            return self._create_fallback_code()

    def execute_code(self, code_content: str) -> PipelineResult:
        """
        Thực thi code và trả về kết quả
        
        Args:
            code_content: Code cần thực thi
            
        Returns:
            PipelineResult với thông tin execution
        """
        self.logger.info("🚀 Đang thực thi code...")
        start_time = datetime.now()
        
        # Tạo temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code_content)
            temp_script_path = temp_file.name
            
        try:
            # Chạy code
            result = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 phút timeout
                cwd=self.output_dir
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                # Thành công
                self.logger.info(f"✅ Code executed successfully! Time: {execution_time:.2f}s")
                
                # Tìm CSV output
                csv_output = self._find_csv_output()
                
                return PipelineResult(
                    success=True,
                    output_file=csv_output or "No CSV found",
                    execution_time=execution_time,
                    statistics={"stdout": result.stdout, "execution_time": execution_time},
                    error_message=None
                )
            else:
                # Lỗi execution
                error_msg = result.stderr or result.stdout
                self.logger.error(f"❌ Execution failed: {error_msg}")
                
                return PipelineResult(
                    success=False,
                    output_file="",
                    execution_time=execution_time,
                    statistics={"stderr": error_msg},
                    error_message=error_msg
                )
                
        except subprocess.TimeoutExpired:
            self.logger.error("⏰ Execution timeout")
            return PipelineResult(
                success=False,
                output_file="",
                execution_time=300,
                statistics={},
                error_message="Code execution timeout"
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Execution error: {e}")
            return PipelineResult(
                success=False,
                output_file="",
                execution_time=execution_time,
                statistics={},
                error_message=str(e)
            )
        finally:
            # Cleanup
            try:
                os.unlink(temp_script_path)
            except:
                pass

    def _find_csv_output(self) -> Optional[str]:
        """Tìm file CSV output mới nhất"""
        csv_files = list(self.output_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
        return None

    def process_pipeline_result(self, code_file_path: str, data_source: str, data_description: str, max_retries: int = 2) -> PipelineResult:
        """
        Main method: đọc code, rewrite, execute
        
        Args:
            code_file_path: Path đến file code cần xử lý
            data_source: Đường dẫn đến thư mục chứa dữ liệu
            data_description: Mô tả cấu trúc thư mục

        Returns:
            PipelineResult với thông tin
        """
        self.logger.info(f"🎯 Processing: {code_file_path}")
        
        try:
            # Đọc original code
            with open(code_file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
                
            self.logger.info(f"📖 Đã đọc {len(original_code)} characters")
            
            current_code = original_code
            
            # Retry loop  
            for attempt in range(max_retries):
                self.logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                # Lưu current code
                attempt_file = self.output_dir / f"attempt_{attempt + 1}_code.py"
                with open(attempt_file, 'w', encoding='utf-8') as f:
                    f.write(current_code)
                
                # Thực thi
                result = self.execute_code(current_code)
                
                if result.success:
                    # Thành công!
                    self.logger.info("🎉 Success!")
                    
                    # Lưu final working code
                    final_file = self.output_dir / "final_working_code.py"
                    with open(final_file, 'w', encoding='utf-8') as f:
                        f.write(current_code)
                    
                    return result
                else:
                    # Failed - rewrite lại với Gemini
                    if attempt < max_retries - 1:
                        self.logger.warning(f"⚠️ Attempt {attempt + 1} failed, rewriting...")
                        current_code = self.rewrite_code_with_gemini(current_code)
                        
                        # Lưu rewritten code
                        rewritten_file = self.output_dir / f"rewritten_{attempt + 1}.py"
                        with open(rewritten_file, 'w', encoding='utf-8') as f:
                            f.write(current_code)
                    else:
                        # Final attempt failed
                        self.logger.error("❌ All attempts failed")
                        return result
            
            # Shouldn't reach here
            return PipelineResult(
                success=False,
                output_file="",
                execution_time=0,
                statistics={},
                error_message="Max retries exceeded"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Process error: {e}")
            return PipelineResult(
                success=False,
                output_file="",
                execution_time=0,
                statistics={},
                error_message=str(e)
            )

# Test function đơn giản
def test_validator():
    """Test validator"""
    validator = ValidatorAgent()
    
    code_file = "outputs/final_solution.py"
    if os.path.exists(code_file):
        result = validator.process_pipeline_result(code_file)
        
        print("\n📊 RESULT:")
        print(f"Success: {result.success}")
        print(f"Output: {result.output_file}")
        print(f"Time: {result.execution_time:.2f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")
    else:
        print(f"❌ File not found: {code_file}")

if __name__ == "__main__":
    test_validator()
