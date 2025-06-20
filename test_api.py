#!/usr/bin/env python3
"""
TEST API CONNECTION - Kiểm tra API key và kết nối Gemini
"""

import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

def test_env_file():
    """Kiểm tra file .env"""
    print("🔍 KIỂM TRA FILE .ENV")
    print("=" * 50)
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ File .env tồn tại")
        print(f"📄 Đường dẫn: {env_file.absolute()}")
        
        # Load env
        load_dotenv()
        
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print(f"✅ GOOGLE_API_KEY found: {api_key[:10]}...{api_key[-10:]}")
            return True
        else:
            print("❌ GOOGLE_API_KEY không tìm thấy trong .env")
            return False
    else:
        print("❌ File .env không tồn tại")
        return False

def test_langchain_import():
    """Test import LangChain"""
    print("\n🔍 KIỂM TRA LANGCHAIN IMPORT")
    print("=" * 50)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ LangChain Google GenAI import thành công")
        return True
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False

def test_simple_api_call():
    """Test API call đơn giản"""
    print("\n🔍 KIỂM TRA API CALL")
    print("=" * 50)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        print("🤖 Đang khởi tạo ChatGoogleGenerativeAI...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.1,
            timeout=30  # 30 second timeout
        )
        print("✅ LLM khởi tạo thành công")
        
        print("🌐 Đang test API call với câu hỏi đơn giản...")
        start_time = time.time()
        
        response = llm.invoke("Hello, please reply with just 'API working'")
        
        api_time = time.time() - start_time
        print(f"✅ API call thành công! Thời gian: {api_time:.2f}s")
        print(f"📄 Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"🐛 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_controller_agent():
    """Test Controller Agent trực tiếp"""
    print("\n🔍 KIỂM TRA CONTROLLER AGENT")
    print("=" * 50)
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from agents.controller_agent import ControllerAgent
        
        print("🎯 Đang khởi tạo Controller Agent...")
        controller = ControllerAgent()
        
        print("🔧 Đang test với problem đơn giản...")
        simple_problem = "Test problem: Classify handwritten digits as prime or non-prime numbers."
        
        start_time = time.time()
        guideline = controller.create_guideline(simple_problem)
        total_time = time.time() - start_time
        
        print(f"✅ Controller Agent test thành công! Thời gian: {total_time:.2f}s")
        print(f"📊 Guideline created with {len(guideline.problem_analysis)} chars analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Controller Agent test failed: {e}")
        print(f"🐛 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Chạy tất cả tests"""
    print("🧪 API & AGENT CONNECTION TEST")
    print("=" * 80)
    
    results = []
    
    # Test 1: .env file
    results.append(("ENV File", test_env_file()))
    
    # Test 2: LangChain import
    results.append(("LangChain Import", test_langchain_import()))
    
    # Test 3: Simple API call
    results.append(("Simple API Call", test_simple_api_call()))
    
    # Test 4: Controller Agent
    results.append(("Controller Agent", test_controller_agent()))
    
    # Summary
    print("\n🎉 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎯 Tất cả tests PASSED! Pipeline sẵn sàng chạy.")
    else:
        print("\n⚠️ Một số tests FAILED. Cần fix trước khi chạy pipeline.")
        
        # Suggestions
        failed_tests = [name for name, success in results if not success]
        print(f"\n💡 Các vấn đề cần fix:")
        for test in failed_tests:
            if test == "ENV File":
                print("   - Tạo file .env với GOOGLE_API_KEY=your_api_key_here")
            elif test == "LangChain Import":
                print("   - pip install langchain-google-genai")
            elif test == "Simple API Call":
                print("   - Kiểm tra API key và kết nối internet")
            elif test == "Controller Agent":
                print("   - Kiểm tra cấu trúc agents/ và models/ folder")

if __name__ == "__main__":
    main() 