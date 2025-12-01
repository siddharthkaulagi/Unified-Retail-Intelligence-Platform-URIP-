# 🤖 AI Chatbot - Retail Forecasting Platform

## Overview
This AI chatbot provides a **unified chat interface** that combines real-time conversations, forecast analysis, and image analysis capabilities using Google's Gemini AI. It's integrated into your existing Retail Sales Forecasting Platform with a modern, ChatGPT-style interface.

## ✨ New Unified Design Features

### 💬 **Modern Chat Interface**
- **Single unified interface** for text and image analysis
- **ChatGPT-style design** with modern UI/UX
- **Fixed input area** at bottom for easy access
- **Real-time conversation flow** with context awareness
- **Auto-generated visualizations** based on AI responses

### 🖼️ **Integrated Image Analysis**
- **Drag & drop or click** to upload images directly in chat
- **Combined text + image analysis** in one seamless experience
- **Preview uploaded images** before sending
- **Support for charts, graphs, and data visualizations**

### 📊 **Smart Visualizations**
- **Auto-generated charts** based on conversation context
- **Interactive Plotly visualizations** for trends and patterns
- **Context-aware visual recommendations**
- **Clean, professional data presentations**

### 🎯 **Enhanced AI Capabilities**
- **Multi-modal analysis** (text + images together)
- **Context-aware responses** with conversation memory
- **Intelligent visualization suggestions**
- **Professional retail forecasting insights**

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- Google Gemini API key
- Required packages (already included in requirements.txt)

### 2. API Key Configuration
The Gemini API key is already configured in your `.env` file:
```
GEMINI_API_KEY=AIzaSyAkBDLLXoTYClUqddXsCWPubvVasRTH98o
```

### 3. Installation
All required packages are already installed. If you need to reinstall:
```bash
pip install -r requirements.txt
```

## How to Use

### Starting the Application
1. Navigate to your project directory:
   ```bash
   cd workspace/streamlit_template
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Access the chatbot:
   - Open your browser and go to `http://localhost:8501`
   - Log in to the application
   - **Click "💬 AI Chat" in the sidebar** of any page (Reports, Dashboard, Demand Forecast, etc.)
   - The chatbot opens as a dedicated page with modern chat interface

### Using the Chatbot

#### 🚀 **Modern Chat Experience**
1. **Type your message** in the input field at the bottom
2. **Upload images** by dragging/dropping or clicking the upload area
3. **Click "🚀 Send"** or press Enter to get AI insights
4. **View auto-generated visualizations** based on your conversation

#### 📱 **Unified Interface Features**
- **Single input area** handles both text and images
- **Fixed bottom input** for easy access (like modern chat apps)
- **Real-time preview** of uploaded images
- **Context-aware visualizations** that appear automatically
- **Clean chat bubbles** with professional styling

#### 🎯 **Quick Actions (Sidebar)**
- **📊 Forecast Analysis** - Quick access to forecasting insights
- **📈 Trend Analysis** - Instant trend analysis
- **Sample Questions** - Pre-built prompts for common queries
- **Clear Chat** - Reset conversation history

#### 💡 **Smart Features**
- **Auto-visualization** - Charts appear based on conversation context
- **Multi-modal analysis** - Text + images analyzed together
- **Conversation memory** - AI remembers context throughout the chat
- **Professional recommendations** - Clean, actionable insights

## Example Use Cases

### Retail Forecasting Questions
```
"What are the current seasonal trends in electronics sales?"
"How should I adjust inventory for the holiday season?"
"What external factors might impact my Q4 sales?"
```

### Image Analysis Examples
- Upload sales charts for trend analysis
- Analyze competitor pricing images
- Extract data from retail reports

### Data Analysis Examples
```
Sales Data Q4 2024:
- Electronics: $150,000 (15% increase)
- Clothing: $89,000 (5% decrease)
- Home Goods: $67,000 (8% increase)
Total: $351,000
```

## API Integration

The chatbot uses the following Gemini AI models:
- **Text Chat**: `gemini-2.0-flash-exp`
- **Image Analysis**: `gemini-2.0-flash-exp` (with vision capabilities)

## Configuration Options

You can customize the chatbot behavior by modifying these `.env` variables:

```env
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
GEMINI_MAX_TOKENS=1500
GEMINI_TEMPERATURE=0.7
GEMINI_TOP_P=0.9
```

## Troubleshooting

### Common Issues

1. **"Google Generative AI package not installed"**
   ```bash
   pip install google-generativeai
   ```

2. **"GEMINI_API_KEY not found"**
   - Check that your `.env` file exists
   - Verify the API key is correctly set
   - Restart the application after updating `.env`

3. **Import errors**
   ```bash
   cd workspace/streamlit_template
   python test_chatbot.py
   ```

### Testing
Run the test script to verify everything is working:
```bash
python test_chatbot.py
```

## File Structure

```
workspace/streamlit_template/
├── utils/
│   └── gemini_ai.py          # Core AI functionality
├── pages/
│   └── 9_🤖_AI_Chatbot.py    # Chatbot interface
├── app.py                    # Main application (updated)
├── test_chatbot.py           # Test script
└── requirements.txt          # Dependencies (updated)
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the test script to diagnose problems
3. Verify your Gemini API key is valid and has quota remaining

## Recent Updates

- ✅ **Unified Chat Interface** - Modern, ChatGPT-style design
- ✅ **Multi-modal Analysis** - Text + images in single interface
- ✅ **Auto-Visualizations** - Context-aware charts and graphs
- ✅ **Fixed Bottom Input** - Easy access like modern chat apps
- ✅ **Professional UI/UX** - Clean, modern styling with gradients
- ✅ **Smart Image Upload** - Drag-drop with preview functionality
- ✅ **Context-Aware Responses** - AI remembers conversation context
- ✅ **Interactive Visualizations** - Plotly charts for data insights
- ✅ **Sidebar Integration** - "💬 AI Chat" button in sidebar of all pages (Reports, Dashboard, Demand Forecast)

---

*Powered by Google's Gemini AI*
*Built for Retail Sales Forecasting Platform*
