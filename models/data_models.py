from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Guideline(BaseModel):
    problem_analysis: str = Field(description="Phân tích bài toán")
    solution_approach: str = Field(description="Cách tiếp cận giải pháp")
    data_requirements: str = Field(description="Yêu cầu dữ liệu")
    pipeline_overview: str = Field(description="Tổng quan pipeline")

class ModelSelection(BaseModel):
    selected_models: List[str] = Field(description="Danh sách model được chọn")
    model_purposes: Dict[str, str] = Field(description="Mục đích từng model")
    model_descriptions: Dict[str, str] = Field(description="Mô tả chi tiết model")
    model_inputs: Dict[str, str] = Field(description="Đầu vào của model")
    model_outputs: Dict[str, str] = Field(description="Đầu ra của model")
    selection_reasoning: str = Field(description="Lý do lựa chọn")

class PreprocessingCode(BaseModel):
    preprocessing_steps: List[str] = Field(description="Các bước preprocessing")
    code_imports: str = Field(description="Các import cần thiết")
    preprocessing_functions: str = Field(description="Hàm preprocessing")
    main_preprocessing_code: str = Field(description="Code preprocessing chính")
    data_validation: str = Field(description="Code validation dữ liệu")

class ProcessingCode(BaseModel):
    model_loading: str = Field(description="Code load models")
    inference_pipeline: str = Field(description="Pipeline inference")
    main_processing_code: str = Field(description="Code processing chính")
    error_handling: str = Field(description="Xử lý lỗi")
    logging_code: str = Field(description="Code logging")

class PostprocessingCode(BaseModel):
    output_formatting: str = Field(description="Format đầu ra")
    result_aggregation: str = Field(description="Tổng hợp kết quả")
    export_code: str = Field(description="Code xuất kết quả")
    final_script: str = Field(description="Script cuối cùng hoàn chỉnh")

class PipelineResult(BaseModel):
    success: bool = Field(description="Pipeline thành công hay không")
    output_file: str = Field(description="Đường dẫn file output CSV")
    execution_time: float = Field(description="Thời gian thực thi")
    statistics: Dict[str, Any] = Field(description="Thống kê kết quả")
    error_message: Optional[str] = Field(description="Thông báo lỗi nếu có")