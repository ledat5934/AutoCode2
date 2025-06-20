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

from models.data_models import Guideline, ModelSelection, PreprocessingCode, ProcessingCode

load_dotenv()

class ProcessingAgent:
    def __init__(self):
        print("⚙️ Processing Agent: Đang khởi tạo...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)
            print("✅ Processing Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Processing Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.parser = PydanticOutputParser(pydantic_object=ProcessingCode)
        print("✅ Processing Agent: Parser khởi tạo thành công")
        
        self.processing_template = ChatPromptTemplate.from_template("""
        Bạn là AI Processing Agent chuyên tạo model inference code cho mọi loại ML task.
        
        PROBLEM GUIDELINE: {guideline}
        SELECTED MODEL: {selected_models}
        PREPROCESSING CODE: {preprocessing_code}
        
        Tạo comprehensive processing code cho ML inference:
        
        1. 🤖 ADAPTIVE MODEL LOADING:
           - Load selected model from HuggingFace URL
           - Load appropriate tokenizers/feature extractors
           - Setup device optimization (CPU/GPU/MPS)
           - Handle model-specific configurations
        
        2. 🔄 INTELLIGENT INFERENCE PIPELINE:
           FOR COMPUTER VISION:
           - Batch image processing
           - Forward pass với vision models
           - Handle different input resolutions
           FOR NLP:
           - Text tokenization và encoding
           - Attention mask handling
           - Sequence classification/generation
           FOR AUDIO:
           - Audio feature extraction
           - Spectogram processing
           - Audio classification/transcription
           FOR TABULAR:
           - Feature preprocessing
           - Numerical prediction
           - Regression/classification output
        
        3. 🎯 MAIN PROCESSING WORKFLOW:
           - Load data từ preprocessing step
           - Initialize appropriate model pipeline
           - Run inference với optimal batch sizes
           - Extract predictions/embeddings/logits
           - Handle confidence scores và uncertainty
           - Post-process outputs theo task requirements
        
        4. ⚠️ ROBUST ERROR HANDLING:
           - Model loading failures
           - OOM (Out of Memory) management
           - Input format mismatches
           - Device compatibility issues
           - Graceful degradation strategies
        
        5. 📊 COMPREHENSIVE LOGGING:
           - Processing progress tracking
           - Model performance metrics
           - Memory usage monitoring
           - Inference time benchmarks
           - Error reporting và debugging
        
        Generate production-ready inference code tự động adapt cho bất kỳ ML domain nào.
        
        IMPORTANT: 
        - Code MUST handle multiple model architectures
        - Include appropriate error recovery mechanisms
        - Response MUST be valid JSON format với properly escaped strings
        
        {format_instructions}
        """)
        print("✅ Processing Agent: Template khởi tạo thành công")

    def clean_json_response(self, raw_response: str) -> str:
        """Clean và fix JSON response từ LLM"""
        print("🧹 Processing: Đang clean JSON response...")
        
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', raw_response)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        
        # Fix common JSON issues in Python code strings
        # Escape unescaped quotes in strings
        lines = cleaned.split('\n')
        fixed_lines = []
        in_string = False
        current_quote = None
        
        for line in lines:
            if '"model_loading":' in line or '"inference_pipeline":' in line or '"main_processing_code":' in line or '"error_handling":' in line or '"logging_code":' in line:
                # This is a code field, need special handling
                # Find the opening quote after the colon
                colon_pos = line.find(':')
                if colon_pos != -1:
                    before_value = line[:colon_pos+1].strip()
                    after_colon = line[colon_pos+1:].strip()
                    
                    if after_colon.startswith('"'):
                        # Extract the string value and properly escape it
                        # Find the closing quote (not escaped)
                        value_start = after_colon.find('"') + 1
                        value_end = len(after_colon)
                        
                        # Find the actual end quote
                        quote_count = 0
                        for i, char in enumerate(after_colon):
                            if char == '"' and (i == 0 or after_colon[i-1] != '\\'):
                                quote_count += 1
                                if quote_count == 2:  # Found closing quote
                                    value_end = i
                                    break
                        
                        if value_end < len(after_colon):
                            value_content = after_colon[value_start:value_end]
                            rest_of_line = after_colon[value_end+1:]
                            
                            # Escape the content properly
                            escaped_content = json.dumps(value_content)[1:-1]  # Remove outer quotes
                            
                            fixed_line = f'{before_value} "{escaped_content}"{rest_of_line}'
                            fixed_lines.append(fixed_line)
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        result = '\n'.join(fixed_lines)
        print(f"✅ Processing: JSON cleaning completed, length: {len(result)}")
        return result

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
            
            print("🌐 Processing: Đang gọi Gemini API...")
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
                print("🧹 Processing: Trying to clean JSON...")
                
                # Try cleaning the JSON
                cleaned_response = self.clean_json_response(response.content)
                
                # Save cleaned response for debugging
                with open("temp/cleaned_processing_response.txt", "w", encoding="utf-8") as f:
                    f.write(cleaned_response)
                
                try:
                    code = self.parser.parse(cleaned_response)
                    print("✅ Processing: Cleaned parse thành công!")
                except Exception as clean_parse_error:
                    print(f"❌ Processing: Cleaned parse also failed: {clean_parse_error}")
                    
                    # Manual JSON extraction as last resort
                    print("🔧 Processing: Attempting manual JSON extraction...")
                    try:
                        # Find JSON content between first { and last }
                        start_brace = response.content.find('{')
                        end_brace = response.content.rfind('}')
                        
                        if start_brace != -1 and end_brace != -1:
                            json_content = response.content[start_brace:end_brace+1]
                            
                            # Try to parse this extracted JSON
                            parsed_json = json.loads(json_content)
                            code = self.parser.parse(json_content)
                            print("✅ Processing: Manual extraction successful!")
                        else:
                            raise Exception("Could not find valid JSON structure")
                    except Exception as manual_error:
                        print(f"❌ Processing: Manual extraction failed: {manual_error}")
                        
                        # Create adaptive fallback based on data type
                        print(f"🔧 Processing: Creating adaptive fallback for {data_type}...")
                        if "Computer Vision" in data_type:
                            code = self._create_cv_processing_fallback()
                        elif "NLP" in data_type:
                            code = self._create_nlp_processing_fallback()
                        elif "Audio" in data_type:
                            code = self._create_audio_processing_fallback()
                        elif "Time Series" in data_type:
                            code = self._create_timeseries_processing_fallback()
                        else:
                            code = self._create_tabular_processing_fallback()
                        
                        print("✅ Processing: Adaptive fallback created")
            
            # Debug output
            print(f"📊 Processing Result Preview:")
            print(f"   🤖 Model Loading: {len(code.model_loading)} characters")
            print(f"   🔄 Inference Pipeline: {len(code.inference_pipeline)} characters")
            print(f"   🎯 Main Processing: {len(code.main_processing_code)} characters")
            print(f"   ⚠️ Error Handling: {len(code.error_handling)} characters")
            print(f"   📊 Logging: {len(code.logging_code)} characters")
            
            print("✅ Processing: Hoàn thành model inference code generation!")
            return code
            
        except Exception as e:
            print(f"❌ Processing: Critical error in processing code generation: {e}")
            print(f"🐛 Processing: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Create adaptive fallback based on detected data type
            print(f"🔧 Processing: Creating critical fallback for {data_type}...")
            if "Computer Vision" in data_type:
                fallback_code = self._create_cv_processing_fallback()
            elif "NLP" in data_type:
                fallback_code = self._create_nlp_processing_fallback()
            elif "Audio" in data_type:
                fallback_code = self._create_audio_processing_fallback()
            elif "Time Series" in data_type:
                fallback_code = self._create_timeseries_processing_fallback()
            else:
                fallback_code = self._create_tabular_processing_fallback()
            
            print("✅ Processing: Critical fallback created")
            return fallback_code

    def _create_cv_processing_fallback(self):
        """Computer Vision processing fallback"""
        return ProcessingCode(
            model_loading="from transformers import AutoImageProcessor, AutoModelForImageClassification\nimport torch\n\nmodel_name = selected_models[0]\nprocessor = AutoImageProcessor.from_pretrained(model_name)\nmodel = AutoModelForImageClassification.from_pretrained(model_name)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel.to(device)",
            inference_pipeline="def process_images(images, model, processor, device):\n    inputs = processor(images, return_tensors='pt').to(device)\n    with torch.no_grad():\n        outputs = model(**inputs)\n        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)\n    return predictions",
            main_processing_code="results = {}\nfor img_path in image_paths:\n    img = load_image(img_path)\n    prediction = process_images([img], model, processor, device)\n    predicted_class = torch.argmax(prediction, dim=-1).item()\n    results[img_path] = predicted_class",
            error_handling="try:\n    # Processing code here\nexcept torch.cuda.OutOfMemoryError:\n    print('GPU memory exceeded, switching to CPU')\n    model.to('cpu')\nexcept Exception as e:\n    print(f'Error processing: {e}')",
            logging_code="import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\nlogger.info(f'Processing {len(image_paths)} images')"
        )

    def _create_nlp_processing_fallback(self):
        """NLP processing fallback"""
        return ProcessingCode(
            model_loading="from transformers import AutoTokenizer, AutoModelForSequenceClassification\nimport torch\n\nmodel_name = selected_models[0]\ntokenizer = AutoTokenizer.from_pretrained(model_name)\nmodel = AutoModelForSequenceClassification.from_pretrained(model_name)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel.to(device)",
            inference_pipeline="def process_texts(texts, model, tokenizer, device, max_length=512):\n    inputs = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)\n    with torch.no_grad():\n        outputs = model(**inputs)\n        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)\n    return predictions",
            main_processing_code="results = {}\nfor text_id, text in enumerate(texts):\n    prediction = process_texts([text], model, tokenizer, device)\n    predicted_class = torch.argmax(prediction, dim=-1).item()\n    results[text_id] = predicted_class",
            error_handling="try:\n    # Processing code here\nexcept torch.cuda.OutOfMemoryError:\n    print('GPU memory exceeded, reducing batch size')\n    batch_size = batch_size // 2\nexcept Exception as e:\n    print(f'Error processing text: {e}')",
            logging_code="import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\nlogger.info(f'Processing {len(texts)} texts')"
        )

    def _create_audio_processing_fallback(self):
        """Audio processing fallback"""
        return ProcessingCode(
            model_loading="from transformers import AutoFeatureExtractor, AutoModelForAudioClassification\nimport torch\n\nmodel_name = selected_models[0]\nfeature_extractor = AutoFeatureExtractor.from_pretrained(model_name)\nmodel = AutoModelForAudioClassification.from_pretrained(model_name)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel.to(device)",
            inference_pipeline="def process_audio(audio_data, model, feature_extractor, device):\n    inputs = feature_extractor(audio_data, return_tensors='pt', sampling_rate=16000).to(device)\n    with torch.no_grad():\n        outputs = model(**inputs)\n        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)\n    return predictions",
            main_processing_code="results = {}\nfor audio_path in audio_paths:\n    audio, sr = librosa.load(audio_path, sr=16000)\n    prediction = process_audio(audio, model, feature_extractor, device)\n    predicted_class = torch.argmax(prediction, dim=-1).item()\n    results[audio_path] = predicted_class",
            error_handling="try:\n    # Processing code here\nexcept Exception as e:\n    print(f'Error processing audio {audio_path}: {e}')\n    continue",
            logging_code="import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\nlogger.info(f'Processing {len(audio_paths)} audio files')"
        )

    def _create_timeseries_processing_fallback(self):
        """Time Series processing fallback"""
        return ProcessingCode(
            model_loading="import torch\nimport torch.nn as nn\nfrom sklearn.preprocessing import StandardScaler\n\n# Load or define time series model\nmodel = torch.load(model_path) if model_path.endswith('.pt') else None\nscaler = StandardScaler()\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            inference_pipeline="def process_sequences(sequences, model, scaler, device):\n    scaled_sequences = scaler.transform(sequences)\n    tensor_sequences = torch.FloatTensor(scaled_sequences).to(device)\n    with torch.no_grad():\n        predictions = model(tensor_sequences)\n    return predictions",
            main_processing_code="results = {}\nfor seq_id, sequence in enumerate(sequences):\n    prediction = process_sequences([sequence], model, scaler, device)\n    results[seq_id] = prediction.cpu().numpy()",
            error_handling="try:\n    # Processing code here\nexcept Exception as e:\n    print(f'Error processing sequence {seq_id}: {e}')\n    continue",
            logging_code="import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\nlogger.info(f'Processing {len(sequences)} time series sequences')"
        )

    def _create_tabular_processing_fallback(self):
        """Tabular processing fallback"""
        return ProcessingCode(
            model_loading="import pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.preprocessing import StandardScaler\n\n# Load or create model\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nscaler = StandardScaler()",
            inference_pipeline="def process_tabular(data, model, scaler):\n    scaled_data = scaler.transform(data)\n    predictions = model.predict(scaled_data)\n    probabilities = model.predict_proba(scaled_data)\n    return predictions, probabilities",
            main_processing_code="results = {}\nX_processed = preprocess_tabular(df)\npredictions, probabilities = process_tabular(X_processed, model, scaler)\nfor i, (pred, prob) in enumerate(zip(predictions, probabilities)):\n    results[i] = {'prediction': pred, 'confidence': np.max(prob)}",
            error_handling="try:\n    # Processing code here\nexcept Exception as e:\n    print(f'Error processing row {i}: {e}')\n    continue",
            logging_code="import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\nlogger.info(f'Processing {len(df)} tabular rows')"
        )