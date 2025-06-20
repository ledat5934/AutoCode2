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

from models.data_models import Guideline, ModelSelection, PreprocessingCode

load_dotenv()

class PreprocessingAgent:
    def __init__(self):
        print("🛠️ Preprocessing Agent: Đang khởi tạo...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)
            print("✅ Preprocessing Agent: LLM khởi tạo thành công")
        except Exception as e:
            print(f"❌ Preprocessing Agent: Lỗi khởi tạo LLM: {e}")
            raise
            
        self.parser = PydanticOutputParser(pydantic_object=PreprocessingCode)
        print("✅ Preprocessing Agent: Parser khởi tạo thành công")
        
        self.preprocessing_template = ChatPromptTemplate.from_template("""
        Bạn là AI Preprocessing Agent chuyên tạo data preprocessing code cho mọi loại ML task.
        
        PROBLEM GUIDELINE: {guideline}
        SELECTED MODEL: {selected_models}
        DATA SOURCE: {data_source}
        
        Tạo comprehensive preprocessing code:
        
        1. 🔍 DATA TYPE DETECTION:
           - Automatically detect data type từ guideline: Images/Text/Audio/CSV/JSON
           - Determine appropriate file formats và extensions
           - Identify data structure và organization
           - Assess data volume và complexity
        
        2. 📦 SMART IMPORTS:
           FOR IMAGES: torch, torchvision, PIL, cv2, albumentations
           FOR TEXT: transformers, tokenizers, datasets, nltk
           FOR AUDIO: librosa, torchaudio, soundfile
           FOR TABULAR: pandas, numpy, sklearn, scipy
           FOR TIME SERIES: pandas, numpy, matplotlib, seaborn
           + ALWAYS: os, pathlib, logging, json
        
        3. 🔄 ADAPTIVE PREPROCESSING FUNCTIONS:
           IMAGES:
           - load_and_resize_image(), normalize_image()
           - data_augmentation(), batch_creator()
           TEXT:
           - tokenize_text(), clean_text(), encode_sequences()
           - handle_special_tokens(), create_attention_masks()
           AUDIO:
           - load_audio(), extract_features(), normalize_audio()
           - spectogram_conversion(), noise_reduction()
           TABULAR:
           - load_csv(), handle_missing_values(), encode_categorical()
           - feature_scaling(), outlier_detection()
        
        4. 🎯 MAIN PREPROCESSING PIPELINE:
           - Auto-detect data format từ source
           - Download/load data từ Google Drive nếu cần
           - Apply appropriate transformations theo data type
           - Create train/validation splits
           - Prepare data theo model input requirements
           - Handle different batch sizes và memory constraints
        
        5. ✅ ROBUST VALIDATION:
           - Check data integrity và format
           - Verify file paths và accessibility
           - Validate tensor shapes và data types
           - Handle edge cases và error scenarios
           - Log processing statistics
        
        Generate production-ready preprocessing code tự động adapt cho bất kỳ data type nào.
        
        IMPORTANT: 
        - Code MUST be adaptive và handle multiple data types
        - Include Google Drive download capability nếu cần
        - Response MUST be valid JSON format
        
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
            
            print("🌐 Preprocessing: Đang gọi Gemini API...")
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
                
                print("🔍 Preprocessing: Đang clean và parse response...")
                
                # Clean JSON content trước khi parse
                cleaned_content = self._clean_json_content(response.content)
                
                code = self.parser.parse(cleaned_content)
                print("✅ Preprocessing: Parse thành công!")
                
                # Debug output
                print(f"📊 Preprocessing Result Preview:")
                print(f"   📦 Imports: {len(code.code_imports)} characters")
                print(f"   🔄 Functions: {len(code.preprocessing_functions)} characters")
                print(f"   🎯 Main Code: {len(code.main_preprocessing_code)} characters")
                print(f"   📋 Steps: {len(code.preprocessing_steps)} steps")
                
                print("✅ Preprocessing: Hoàn thành code generation!")
                return code
            else:
                print("❌ Preprocessing: Response is None or empty!")
                raise Exception("Empty response from Gemini API")
                
        except Exception as e:
            print(f"❌ Preprocessing: Lỗi trong quá trình tạo preprocessing code: {e}")
            print(f"🐛 Preprocessing: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Create adaptive fallback response based on detected data type
            print(f"🔧 Preprocessing: Creating adaptive fallback for {data_type}...")
            
            if "Computer Vision" in data_type:
                fallback_code = self._create_cv_fallback()
            elif "NLP" in data_type:
                fallback_code = self._create_nlp_fallback()
            elif "Audio" in data_type:
                fallback_code = self._create_audio_fallback()
            elif "Time Series" in data_type:
                fallback_code = self._create_timeseries_fallback()
            else:
                fallback_code = self._create_tabular_fallback()
            
            print("✅ Preprocessing: Adaptive fallback response created")
            return fallback_code

    def _clean_json_content(self, content: str) -> str:
        """Clean JSON content để fix escape character issues"""
        try:
            # Extract JSON từ markdown code blocks nếu có
            if "```json" in content:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            # Fix common escape issues trong regex patterns
            import json
            import re
            
            # Tìm và fix regex patterns với backslash
            content = re.sub(r'\\w\\s', r'\\\\w\\\\s', content)  # Fix \w\s
            content = re.sub(r'\\w', r'\\\\w', content)  # Fix \w 
            content = re.sub(r'\\s', r'\\\\s', content)  # Fix \s
            content = re.sub(r'\\d', r'\\\\d', content)  # Fix \d
            content = re.sub(r'\\n', r'\\\\n', content)  # Fix \n
            content = re.sub(r'\\t', r'\\\\t', content)  # Fix \t
            content = re.sub(r'\\r', r'\\\\r', content)  # Fix \r
            
            # Fix file paths với backslash
            content = re.sub(r'"[^"]*\\[^"]*"', lambda m: m.group(0).replace('\\', '\\\\'), content)
            
            print("✅ JSON content cleaned successfully")
            return content
            
        except Exception as e:
            print(f"⚠️ Error cleaning JSON content: {e}")
            return content

    def _create_cv_fallback(self):
        """Computer Vision fallback"""
        return PreprocessingCode(
            preprocessing_steps=[
                "Download images từ Google Drive",
                "Load và validate image formats",
                "Resize images theo model requirements",
                "Convert to tensors",
                "Apply normalization và augmentation",
                "Create train/validation splits"
            ],
            code_imports="import torch\nimport torchvision.transforms as transforms\nfrom PIL import Image\nimport pandas as pd\nimport numpy as np\nimport os\nimport glob\nfrom pathlib import Path\nimport gdown",
            preprocessing_functions="def load_image(path):\n    img = Image.open(path).convert('RGB')\n    return img\n\ndef download_from_drive(drive_url, output_path):\n    gdown.download_folder(drive_url, output=output_path, quiet=False)",
            main_preprocessing_code="# Download and preprocess images\ndownload_from_drive(drive_url, 'data/')\nimages = glob.glob('data/**/*.jpg', recursive=True)\nprocessed_images = [load_image(img) for img in images]",
            data_validation="assert all(isinstance(img, Image.Image) for img in processed_images)"
        )

    def _create_nlp_fallback(self):
        """NLP fallback"""
        return PreprocessingCode(
            preprocessing_steps=[
                "Download text data từ Google Drive",
                "Load và clean text files",
                "Tokenize text sequences",
                "Create attention masks",
                "Encode labels",
                "Prepare datasets"
            ],
            code_imports="from transformers import AutoTokenizer\nimport pandas as pd\nimport numpy as np\nimport os\nfrom pathlib import Path\nimport gdown\nimport json",
            preprocessing_functions="def clean_text(text):\n    return text.strip().lower()\n\ndef tokenize_texts(texts, tokenizer, max_length=512):\n    return tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')",
            main_preprocessing_code="# Download and preprocess text data\ndownload_from_drive(drive_url, 'data/')\ntexts = pd.read_csv('data/texts.csv')\ntokenizer = AutoTokenizer.from_pretrained(model_name)\nencoded = tokenize_texts(texts['text'].tolist(), tokenizer)",
            data_validation="assert 'input_ids' in encoded and 'attention_mask' in encoded"
        )

    def _create_audio_fallback(self):
        """Audio fallback"""
        return PreprocessingCode(
            preprocessing_steps=[
                "Download audio files từ Google Drive",
                "Load audio với librosa",
                "Extract spectogram features",
                "Normalize audio data",
                "Apply audio augmentation",
                "Create train/test splits"
            ],
            code_imports="import librosa\nimport numpy as np\nimport pandas as pd\nimport os\nfrom pathlib import Path\nimport gdown\nimport soundfile as sf",
            preprocessing_functions="def load_audio(path, sr=22050):\n    audio, _ = librosa.load(path, sr=sr)\n    return audio\n\ndef extract_features(audio):\n    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)\n    return mfcc",
            main_preprocessing_code="# Download and preprocess audio\ndownload_from_drive(drive_url, 'data/')\naudio_files = glob.glob('data/**/*.wav', recursive=True)\nfeatures = [extract_features(load_audio(f)) for f in audio_files]",
            data_validation="assert all(f.shape[0] == 13 for f in features)  # MFCC features"
        )

    def _create_timeseries_fallback(self):
        """Time Series fallback"""
        return PreprocessingCode(
            preprocessing_steps=[
                "Download time series data từ Google Drive",
                "Load CSV/JSON data",
                "Handle missing values",
                "Create time-based features",
                "Apply scaling/normalization",
                "Create sequences for training"
            ],
            code_imports="import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\nimport os\nfrom pathlib import Path\nimport gdown",
            preprocessing_functions="def load_timeseries(path):\n    df = pd.read_csv(path, parse_dates=['timestamp'])\n    return df\n\ndef create_sequences(data, seq_length=10):\n    sequences = []\n    for i in range(len(data) - seq_length):\n        sequences.append(data[i:i+seq_length])\n    return np.array(sequences)",
            main_preprocessing_code="# Download and preprocess time series\ndownload_from_drive(drive_url, 'data/')\ndf = load_timeseries('data/timeseries.csv')\nsequences = create_sequences(df.values)",
            data_validation="assert sequences.shape[1] == 10  # sequence length"
        )

    def _create_tabular_fallback(self):
        """Tabular fallback"""
        return PreprocessingCode(
            preprocessing_steps=[
                "Download tabular data từ Google Drive",
                "Load CSV/Excel files",
                "Handle missing values",
                "Encode categorical variables",
                "Apply feature scaling",
                "Create train/test splits"
            ],
            code_imports="import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.model_selection import train_test_split\nimport os\nfrom pathlib import Path\nimport gdown",
            preprocessing_functions="def load_tabular(path):\n    df = pd.read_csv(path)\n    return df\n\ndef preprocess_tabular(df):\n    # Handle missing values\n    df = df.dropna()\n    # Encode categorical\n    for col in df.select_dtypes(include=['object']).columns:\n        le = LabelEncoder()\n        df[col] = le.fit_transform(df[col])\n    return df",
            main_preprocessing_code="# Download and preprocess tabular data\ndownload_from_drive(drive_url, 'data/')\ndf = load_tabular('data/dataset.csv')\nprocessed_df = preprocess_tabular(df)",
            data_validation="assert processed_df.isnull().sum().sum() == 0  # No missing values"
        )