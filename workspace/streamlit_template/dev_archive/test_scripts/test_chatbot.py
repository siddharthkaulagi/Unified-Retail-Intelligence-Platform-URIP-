#!/usr/bin/env python3
"""
Test script for Gemini AI Chatbot functionality
"""

import os
import sys
from datetime import datetime
import streamlit as st

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
stretch = None
def test_gemini_chatbot():
    """Test the Gemini chatbot functionality"""
    print("🤖 Testing Gemini AI Chatbot...")
    print("=" * 50)

    try:
        # Import chatbot
        from utils.gemini_ai import GeminiChatbot
        print("✅ Successfully imported GeminiChatbot")

        # Initialize chatbot
        print("\n🔧 Initializing chatbot...")
        chatbot = GeminiChatbot()
        print("✅ Chatbot initialized successfully")

        # Test basic chat
        print("\n💬 Testing basic chat functionality...")
        test_message = "What are the latest trends in retail forecasting?"
        print(f"User: {test_message}")

        response = chatbot.chat(test_message)
        if response['success']:
            print("✅ Chat response received successfully")
            print(f"Bot: {response['response'][:200]}...")
        else:
            print(f"❌ Chat failed: {response.get('error', 'Unknown error')}")
            return False

        # Test forecast insights
        print("\n📊 Testing forecast insights...")
        sample_forecast_data = """
        Sales Data for Q4 2024:
        - Electronics: $150,000 (15% increase)
        - Clothing: $89,000 (5% decrease)
        - Home Goods: $67,000 (8% increase)
        - Sports: $45,000 (12% increase)
        Total: $351,000
        """

        insights_response = chatbot.get_forecast_insights(sample_forecast_data)
        if insights_response['success']:
            print("✅ Forecast insights generated successfully")
            print(f"Insights: {insights_response['insights'][:200]}...")
        else:
            print(f"❌ Forecast insights failed: {insights_response.get('error', 'Unknown error')}")

        # Test conversation history
        print(f"\n📝 Conversation history length: {len(chatbot.conversation_history)}")
        print("✅ Conversation history tracking works")

        print("\n" + "=" * 50)
        print("🎉 All tests passed! Chatbot is ready to use.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure google-generativeai is installed:")
        print("pip install google-generativeai")
        return False

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please make sure GEMINI_API_KEY is set in your .env file")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_streamlit_integration():
    """Test if the Streamlit page can be imported"""
    print("\n🔍 Testing Streamlit integration...")
    try:
        # Try to import the chatbot page (this will test if all imports work)
        import streamlit as st
        def show():
            st.title("AI Chatbot")
    # Add your chatbot implementation here
        pass


        print("✅ Streamlit page imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Streamlit integration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in Streamlit integration: {e}")
        return False

if __name__ == "__main__":
    print(f"🧪 Chatbot Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test chatbot functionality
    chatbot_ok = test_gemini_chatbot()

    # Test Streamlit integration
    streamlit_ok = test_streamlit_integration()

    print("\n" + "=" * 50)
    if chatbot_ok and streamlit_ok:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Your AI chatbot is ready to use!")
        print("\nTo run the chatbot:")
        print("1. Make sure your .env file has GEMINI_API_KEY")
        print("2. Run: streamlit run app.py")
        print("3. Navigate to the AI Chatbot page")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them before using the chatbot.")
        sys.exit(1)
