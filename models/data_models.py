from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Guideline(BaseModel): # For controller_agent
    problem_analysis: str = Field(description="The problem analysis")
    solution_approach: str = Field(description="The solution approach")
    data_requirements: str = Field(description="The data requirements")
    pipeline_overview: str = Field(description="The pipeline overview(Let the selector choose appropriate model from model pool which can be used, preprocessing,call model, postprocessing, etc., )")

class ModelSelection(BaseModel):
    selected_models: List[str] = Field(description="The list of selected model from model pool which can be used to solve problem )")
    model_purpose: Dict[str, str] = Field(description="The purpose of each selected model(what each of them can do)")
    model_description: Dict[str, str] = Field(description="The description of each selected model(Data pretrained on, Domain, etc., )")
    model_input: Dict[str, str] = Field(description="The input of each selected model")
    model_output: Dict[str, str] = Field(description="The output of each selected model")
    selection_reasoning: str = Field(description="The reason for selecting the model")

class PreprocessingCode(BaseModel):
    preprocessing_steps: List[str] = Field(description="The list of preprocessing steps")
    code_imports: str = Field(description="Import neccessary libraries")
    preprocessing_functions: List[str] = Field(description="The list of function use for preprocessing")
    main_function: str = Field(description="The main function for preprocessing using function")

class ProcessingCode(BaseModel):
    model_loading: str = Field(description="The model loading code(directly load model by using URL from model pool, don't load submodel)")
    inference_pipeline: str = Field(description="Pipeline inference")
    main_processing_code: str = Field(description="Code processing chính")

class PostprocessingCode(BaseModel):
    output_format: str = Field(description="The output format of the problem")
    result_aggregation: str = Field(description="Aggregation of result from model")
    export_code: str = Field(description="Code to export result(to csv, to json, to excel, etc., )")
    final_script: str = Field(description="Final script to run the code")

class PipelineResult(BaseModel):
    success: bool = Field(description="Pipeline thành công hay không")
    output_file: str = Field(description="Đường dẫn file output CSV")
    execution_time: float = Field(description="Thời gian thực thi")
    statistics: Dict[str, Any] = Field(description="Thống kê kết quả")
    error_message: Optional[str] = Field(description="Thông báo lỗi nếu có")

