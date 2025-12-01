#!/usr/bin/env python3
"""Test Gemini API key functionality"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("ERROR: No GEMINI_API_KEY found in environment")
    exit(1)

print(f"SUCCESS: Found API key: {api_key[:20]}...")

# Test import
try:
    from utils.gemini_ai import GeminiChatbot, GeminiAIAssistant
    print("SUCCESS: Gemini AI classes imported successfully")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    exit(1)

# Test chatbot initialization
try:
    chatbot = GeminiChatbot()
    print("SUCCESS: GeminiChatbot initialized with API key")
except Exception as e:
    print(f"ERROR: Chatbot initialization failed: {e}")
    print("This might be expected if the API key is invalid or quota exceeded")

print("API key configuration test completed!")
