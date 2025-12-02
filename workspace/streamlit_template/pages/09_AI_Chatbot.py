# workspace/streamlit_template/pages/09_AI_Chatbot.py

import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import sys

# allow importing utils from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.ui_components import render_sidebar as render_global_sidebar
from utils.load_css import load_css_for_page   # robust shared loader

# -----------------------------------------------------------
# IMPORTANT: set_page_config must be the FIRST Streamlit call
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Chatbot - Retail Forecasting",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS file (robust, resolves relative to this file)
load_css_for_page(__file__)

# Custom inline CSS for modern chat interface (still okay after set_page_config)
st.markdown("""
<style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 70%;
        float: right;
        clear: both;
        word-wrap: break-word;
    }
    .bot-message {
        background-color: #e9ecef;
        color: black;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 70%;
        float: left;
        clear: both;
        word-wrap: break-word;
    }
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: white;
        padding: 20px;
        border-radius: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .file-upload-area {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        background-color: #fafafa;
    }
    .chat-input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 12px 20px;
        font-size: 16px;
        width: 100%;
        margin-bottom: 10px;
    }
    .send-button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
        float: right;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Import Gemini AI classes after set_page_config so Streamlit calls are allowed here
try:
    from utils.gemini_ai import GeminiChatbot, GeminiAIAssistant
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    # Defer user-facing errors to runtime (don't call st.error at import-time outside app flow)
    # We'll show a warning during initialization if Gemini isn't available.


# Constants
MAX_CHAT_HISTORY = 100
MAX_IMAGE_SIZE = (800, 600)
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'gif', 'bmp']
VISUALIZATION_TYPES = ["forecast_trends", "category_comparison", "seasonal_pattern"]


def initialize_chatbot() -> bool:
    """Initialize the chatbot if not already done"""
    if 'chatbot' not in st.session_state:
        if not GEMINI_AVAILABLE:
            st.warning("⚠️ Gemini AI not available. Core forecasting features will work without AI assistance.")
            st.session_state.chatbot = None
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.session_state.show_image_upload = False
            return False

        try:
            st.session_state.chatbot = GeminiChatbot()
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.session_state.show_image_upload = False
            return True
        except Exception as e:
            st.warning(f"⚠️ AI Chatbot unavailable due to API quota limits. Core forecasting features will work without AI assistance.")
            st.session_state.chatbot = None
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.session_state.show_image_upload = False
            return False
    return st.session_state.chatbot is not None


def validate_image_file(uploaded_file) -> Tuple[bool, Optional[Image.Image], str]:
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, None, "No file uploaded"

    try:
        image = Image.open(uploaded_file)

        # Check file size (limit to 10MB)
        if hasattr(uploaded_file, 'size') and uploaded_file.size > 10 * 1024 * 1024:
            return False, None, "File size too large (max 10MB)"

        # Resize if too large
        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        return True, image, "Image uploaded successfully"

    except Exception as e:
        return False, None, f"Error processing image: {str(e)}"


def manage_chat_history(chat_history: List[Dict]) -> List[Dict]:
    """Manage chat history size and cleanup"""
    if len(chat_history) > MAX_CHAT_HISTORY:
        # Keep only the last N messages
        return chat_history[-MAX_CHAT_HISTORY:]

    return chat_history


def display_chat_message(message: str, is_user: bool = True, image: Optional[Image.Image] = None) -> None:
    """Display a chat message in the UI"""
    if is_user:
        col1, col2 = st.columns([1, 3])
        with col2:
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong><br>{message}
            </div>
            """, unsafe_allow_html=True)
            if image:
                st.image(image, width=200)
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 AI Assistant:</strong><br>{message}
            </div>
            """, unsafe_allow_html=True)


@lru_cache(maxsize=10)
def get_sample_data(data_type: str) -> Optional[pd.DataFrame]:
    """Get sample data for visualizations with caching"""
    try:
        if data_type == "forecast_trends":
            dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
            values = [100 + i*5 + (i**2) for i in range(12)]
            return pd.DataFrame({'date': dates, 'value': values})

        elif data_type == "category_comparison":
            categories = ['Electronics', 'Clothing', 'Home', 'Sports']
            values = [45000, 32000, 28000, 15000]
            return pd.DataFrame({'category': categories, 'value': values})

        elif data_type == "seasonal_pattern":
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            seasonal = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.1, 1.0, 1.1, 1.2]
            return pd.DataFrame({'month': months, 'seasonal_factor': seasonal})

    except Exception as e:
        st.error(f"Error generating sample data: {e}")
    return None


def create_visualization(data_type: str, data: Optional[pd.DataFrame] = None) -> Optional[go.Figure]:
    """Create visualizations based on data type with improved error handling"""
    try:
        sample_data = data or get_sample_data(data_type)
        if sample_data is None:
            return None

        if data_type == "forecast_trends":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_data['date'],
                y=sample_data['value'],
                mode='lines+markers',
                name='Forecast Trend',
                line=dict(color='#007bff', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="📈 Sales Forecast Trend",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                template="plotly_white",
                height=400
            )
            return fig

        elif data_type == "category_comparison":
            fig = px.bar(
                sample_data,
                x='category',
                y='value',
                title="📊 Sales by Category",
                color='category',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                template="plotly_white",
                height=400,
                showlegend=False
            )
            return fig

        elif data_type == "seasonal_pattern":
            fig = px.line(
                sample_data,
                x='month',
                y='seasonal_factor',
                title="🌊 Seasonal Sales Pattern",
                markers=True
            )
            fig.update_layout(
                template="plotly_white",
                height=400,
                yaxis_title="Seasonal Factor"
            )
            return fig

    except Exception as e:
        st.error(f"❌ Error creating visualization: {e}")
    return None


def get_fallback_response(message: str) -> str:
    """Provide fallback responses when AI is not available"""
    message_lower = message.lower()

    # Common retail forecasting questions
    if any(word in message_lower for word in ['trend', 'latest', 'current']):
        response = """
        📈 **Current Retail Trends (Based on Industry Knowledge):**

        **E-commerce Growth:** Online retail continues to grow at 10-15% annually, with mobile commerce leading at 25%+ growth.

        **Sustainability Focus:** Eco-friendly products and sustainable packaging are increasingly important to consumers.

        **Personalization:** AI-driven personalized shopping experiences are becoming standard.

        **Omnichannel Shopping:** Seamless integration between online and offline shopping experiences.

        **Local & Fresh:** Demand for locally sourced and fresh products remains strong.

        *💡 Tip: Upload your sales data to get specific trend analysis for your business.*
    """
    elif any(word in message_lower for word in ['forecast', 'predict', 'future']):
        response = """
        🔮 **Sales Forecasting Best Practices:**

        **Key Factors to Consider:**
        • Historical sales data (2+ years recommended)
        • Seasonal patterns and holidays
        • Economic indicators (inflation, unemployment)
        • Competitive landscape
        • Marketing campaigns and promotions

        **Recommended Models:**
        • ARIMA/SARIMA for seasonal data
        • XGBoost/LightGBM for complex patterns
        • Prophet for holiday and event effects

        **Accuracy Tips:**
        • Use multiple models and ensemble them
        • Regularly retrain models with new data
        • Monitor forecast accuracy metrics

        *📊 Use the Model Selection page to try different forecasting algorithms.*
        """
    elif any(word in message_lower for word in ['seasonal', 'pattern', 'cycle']):
        response = """
        🌊 **Seasonal Sales Patterns:**

        **Common Retail Seasons:**
        • **Q4 (Oct-Dec):** Peak season with holiday shopping
        • **Q1 (Jan-Mar):** Post-holiday slump, focus on clearance
        • **Q2 (Apr-Jun):** Spring renewal, tax refund spending
        • **Q3 (Jul-Sep):** Back-to-school, summer promotions

        **Weekly Patterns:**
        • Weekends typically show 20-30% higher sales
        • Monday-Wednesday: Steady baseline sales
        • Thursday-Sunday: Peak shopping days

        **Monthly Variations:**
        • End-of-month: Salary payment effects
        • Mid-month: Typically lower sales
        • Payday cycles vary by region

        *📈 Check the Dashboard page to visualize seasonal patterns in your data.*
        """
    elif any(word in message_lower for word in ['inventory', 'stock', 'supply']):
        response = """
        📦 **Inventory Management Strategies:**

        **Key Metrics to Track:**
        • Inventory turnover ratio (ideal: 4-6 times/year)
        • Stock-out rate (<5% recommended)
        • Overstock percentage (<10% recommended)
        • Carrying costs vs. stockout costs

        **Optimization Techniques:**
        • ABC analysis (categorize by value/volume)
        • Safety stock calculations
        • Economic order quantity (EOQ)
        • Just-in-time inventory for fast-moving items

        **Seasonal Considerations:**
        • Build inventory before peak seasons
        • Liquidate excess stock strategically
        • Use data-driven reorder points

        *⚙️ Visit the Settings page to configure inventory parameters.*
        """
    elif any(word in message_lower for word in ['accuracy', 'improve', 'better']):
        response = """
        🎯 **Improving Forecast Accuracy:**

        **Data Quality Improvements:**
        • Clean and validate historical data
        • Include external factors (weather, events, economy)
        • Capture promotional activities accurately
        • Regular data audits and corrections

        **Model Enhancements:**
        • Use ensemble methods (combine multiple models)
        • Implement machine learning techniques
        • Consider hierarchical forecasting for product categories
        • Regular model retraining and validation

        **Process Improvements:**
        • Collaborative forecasting with stakeholders
        • Regular forecast reviews and adjustments
        • Track forecast accuracy metrics weekly
        • Learn from forecast errors

        *📊 The Dashboard shows forecast accuracy metrics for your models.*
        """
    else:
        response = f"""
        🤖 **AI Assistant Currently Unavailable**

        I'm currently operating in offline mode due to API service limits. However, I can still help you with retail forecasting using our core features:

        **Available Features:**
        • 📊 Data upload and preprocessing
        • 🔮 Multiple forecasting models (ARIMA, XGBoost, LightGBM, etc.)
        • 📈 Interactive dashboards and visualizations
        • 📋 Automated report generation
        • 📊 CRM analytics and customer insights

        **Quick Actions:**
        • Upload your sales data on the Data Upload page
        • Try different forecasting models on the Model Selection page
        • View results on the Dashboard page
        • Generate reports on the Reports page

        Your question: "*{message}*"

        💡 **Tip:** The AI features will be restored once API limits are reset. In the meantime, explore the forecasting models available in the application!
        """
    return response


def detect_visualization_type(message: str) -> Optional[str]:
    """Detect which type of visualization to show based on message content"""
    message_lower = message.lower()

    if any(word in message_lower for word in ['trend', 'forecast', 'prediction', 'future']):
        return "forecast_trends"
    elif any(word in message_lower for word in ['category', 'breakdown', 'segment', 'comparison']):
        return "category_comparison"
    elif any(word in message_lower for word in ['seasonal', 'pattern', 'cycle', 'monthly']):
        return "seasonal_pattern"

    return None


def render_local_actions() -> None:
    """Render the sidebar with quick actions and sample questions"""
    st.markdown("### 🚀 Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Forecast\nAnalysis", key="qa_forecast"):
            st.session_state.quick_action_message = "Analyze my recent forecast data and provide insights"
            st.rerun()
    with col2:
        if st.button("📈 Trend\nAnalysis", key="qa_trend"):
            st.session_state.quick_action_message = "Show me current retail sales trends"
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")

    sample_questions = [
        "What are the latest retail trends?",
        "How to improve forecast accuracy?",
        "Analyze seasonal patterns",
        "Best inventory strategies"
    ]

    for question in sample_questions:
        if st.button(question, key=f"sample_{question[:10]}"):
            st.session_state.quick_action_message = question
            st.rerun()

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", type="secondary", key="clear_chat"):
        st.session_state.chat_history = []
        if 'chatbot' in st.session_state and st.session_state.chatbot:
            try:
                st.session_state.chatbot.clear_conversation()
            except Exception:
                pass
        st.rerun()

    st.caption(f"💬 Messages: {len(st.session_state.chat_history) if 'chat_history' in st.session_state else 0}")


def render_chat_history() -> None:
    """Render the chat history with messages and visualizations"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if st.session_state.get('chat_history'):
        for i, chat_item in enumerate(st.session_state.chat_history):
            display_chat_message(chat_item['user'], True, chat_item.get('image'))
            display_chat_message(chat_item['bot'], is_user=False)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px; color: #666;'>
            <h3>👋 Welcome to AI Retail Assistant!</h3>
            <p>Ask me anything about retail forecasting, sales trends, or upload images for analysis.</p>
            <p><em>Try: "What are the current retail trends?" or upload a sales chart</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_input_area() -> Tuple[str, bool]:
    """Render the input area with file upload and chat input"""
    # Chat input
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        chat_input = st.text_input(
            "Ask me anything about retail forecasting...",
            key="chat_input",
            placeholder="e.g., What are the current trends in retail sales? Or upload an image...",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("🚀 Send", key="send_button", type="primary")

    with col3:
        if st.button("🖼️", key="upload_button", help="Upload image for analysis"):
            st.session_state.show_image_upload = True

    return chat_input, send_button


def handle_chat_input(chat_input: str, send_button: bool) -> None:
    """Handle chat input and AI responses"""
    # Check for quick action message first
    if 'quick_action_message' in st.session_state and st.session_state.quick_action_message:
        chat_input = st.session_state.quick_action_message
        st.session_state.quick_action_message = None
        send_button = True

    if send_button and chat_input and chat_input.strip():
        # Manage chat history size
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history = manage_chat_history(st.session_state.chat_history)

        # Add user message to history
        chat_message = {
            'user': chat_input,
            'bot': '🤔 Analyzing your question...',
            'timestamp': datetime.now().isoformat(),
            'image': st.session_state.get('uploaded_image')
        }
        st.session_state.chat_history.append(chat_message)

        # Get AI response or fallback
        with st.spinner("🤖 Thinking..."):
            try:
                if st.session_state.get('chatbot') is None:
                    # Fallback responses when AI is not available
                    fallback_response = get_fallback_response(chat_input)
                    st.session_state.chat_history[-1]['bot'] = fallback_response
                elif st.session_state.get('uploaded_image'):
                    # If there's an uploaded image, analyze it with the text
                    response = st.session_state.chatbot.analyze_image(
                        st.session_state.uploaded_image,
                        chat_input
                    )
                    st.session_state.uploaded_image = None

                    if response and response.get('success', False):
                        st.session_state.chat_history[-1]['bot'] = response.get('analysis', 'Image analysis completed')
                    else:
                        error_msg = response.get('error', 'Unknown error') if response else 'No response from AI'
                        st.session_state.chat_history[-1]['bot'] = f"❌ Error: {error_msg}"
                else:
                    response = st.session_state.chatbot.chat(chat_input)

                    if response and (response.get('success', False) or response.get('response')):
                        # Update the last message with actual response
                        st.session_state.chat_history[-1]['bot'] = response.get('analysis', response.get('response', 'No response'))
                    else:
                        error_msg = response.get('error', 'Unknown error') if response else 'No response from AI'
                        st.session_state.chat_history[-1]['bot'] = f"❌ Error: {error_msg}"

            except Exception as e:
                st.session_state.chat_history[-1]['bot'] = f"❌ Error: {str(e)}"

        st.rerun()


def handle_image_upload() -> None:
    """Handle image upload and validation"""
    if not st.session_state.get('show_image_upload', False):
        return

    uploaded_file = st.file_uploader(
        label="Choose an image file",
        type=SUPPORTED_IMAGE_TYPES,
        help="Upload charts, graphs, or any retail-related images for analysis",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        is_valid, image, message = validate_image_file(uploaded_file)

        if is_valid and image:
            st.session_state.uploaded_image = image
            # Show preview
            st.markdown("**📷 Image Preview:**")
            st.image(image, width=300)
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")


def render_auto_scroll() -> None:
    """Render JavaScript for auto-scrolling chat"""
    st.markdown("""
    <script>
        function scrollToBottom() {
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        // Scroll when new messages are added
        const observer = new MutationObserver(scrollToBottom);
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }

        window.onload = scrollToBottom;
    </script>
    """, unsafe_allow_html=True)


def render_ai_analyzer():
    """Render the AI Analyzer section for document analysis"""
    st.markdown("### 📄 AI Document Analyzer")
    st.markdown("Upload forecast reports or documents for AI-powered analysis and insights")

    # File upload section
    uploaded_file = st.file_uploader(
        "📄 Upload Report/Document",
        type=['docx', 'pdf', 'txt', 'csv', 'xlsx'],
        help="Upload forecast reports, sales data, or any business documents for AI analysis"
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # File info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.metric("File Type", uploaded_file.type or "Unknown")
        with col3:
            if st.button("🗑️ Clear File", key="clear_file"):
                st.rerun()

        # Analysis options
        st.markdown("#### 🔍 Analysis Options")

        analysis_type = st.selectbox(
            "Analysis Type",
            [
                " 📊 Comprehensive Business Analysis",
                " 📈 Sales Trend Analysis",
                " 📦 Inventory Optimization",
                " 🎯 Performance Insights",
                " 🔮 Predictive Insights",
                " 💼 Strategic Recommendations"
            ]
        )

        include_visualizations = st.checkbox("Include Data Visualizations", value=True)
        analysis_depth = st.selectbox("Analysis Depth", ["Quick Overview", "Detailed Analysis", "Comprehensive Report"])

        if st.button("🚀 Analyze Document", type="primary", key="analyze_doc"):
            if not GEMINI_AVAILABLE or st.session_state.get('chatbot') is None:
                st.warning("⚠️ AI Document Analysis is currently unavailable due to API quota limits. Core forecasting features are still available.")
                st.info("💡 **Alternative:** Use the core forecasting models on the Model Selection page to analyze your data.")
                return

            with st.spinner("🤖 Analyzing document with AI..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Read file content based on type
                    if uploaded_file.name.endswith('.docx'):
                        try:
                            from docx import Document
                            doc = Document(tmp_file_path)
                            file_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                        except Exception:
                            file_content = "Error reading Word document"
                    elif uploaded_file.name.endswith('.pdf'):
                        try:
                            import PyPDF2
                            with open(tmp_file_path, 'rb') as pdf_file:
                                pdf_reader = PyPDF2.PdfReader(pdf_file)
                                file_content = '\n'.join([page.extract_text() for page in pdf_reader.pages])
                        except Exception:
                            file_content = "Error reading PDF document"
                    elif uploaded_file.name.endswith('.csv'):
                        try:
                            df = pd.read_csv(tmp_file_path)
                            file_content = f"CSV Data with {len(df)} rows and {len(df.columns)} columns:\n\n"
                            file_content += str(df.head(10)) + "\n\nColumn info:\n" + str(df.dtypes)
                        except Exception:
                            file_content = "Error reading CSV file"
                    elif uploaded_file.name.endswith('.xlsx'):
                        try:
                            df = pd.read_excel(tmp_file_path)
                            file_content = f"Excel Data with {len(df)} rows and {len(df.columns)} columns:\n\n"
                            file_content += str(df.head(10)) + "\n\nColumn info:\n" + str(df.dtypes)
                        except Exception:
                            file_content = "Error reading Excel file"
                    else:
                        file_content = "Text file content"

                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except Exception:
                        pass

                    # Perform AI analysis
                    if file_content and len(file_content.strip()) > 0:
                        # Create context-aware prompt based on analysis type
                        analysis_prompts = {
                            " 📊 Comprehensive Business Analysis": "Provide a comprehensive business analysis including market trends, competitive insights, and strategic recommendations.",
                            " 📈 Sales Trend Analysis": "Analyze sales trends, patterns, and forecasting accuracy from the data.",
                            " 📦 Inventory Optimization": "Focus on inventory management, stock optimization, and supply chain recommendations.",
                            " 🎯 Performance Insights": "Provide performance insights, KPIs analysis, and improvement recommendations.",
                            " 🔮 Predictive Insights": "Generate predictive insights and future trend analysis.",
                            " 💼 Strategic Recommendations": "Focus on strategic business recommendations and long-term planning."
                        }

                        base_prompt = analysis_prompts.get(analysis_type, "Provide comprehensive analysis of this document.")

                        full_prompt = f"""
                        You are a senior retail analytics consultant. Please analyze the following document and provide detailed insights.

                        DOCUMENT CONTENT:
                        {file_content}

                        ANALYSIS REQUIREMENTS:
                        {base_prompt}

                        DEPTH: {analysis_depth}

                        Please provide:
                        1. Executive Summary
                        2. Key Findings
                        3. Detailed Analysis
                        4. Actionable Recommendations
                        5. Risk Assessment (if applicable)

                        Format your response in a clear, professional manner suitable for business executives.
                        """

                        # Get AI response using Gemini
                        response = st.session_state.chatbot.chat(full_prompt)

                        # Store analysis result
                        st.session_state.document_analysis = {
                            'file_name': uploaded_file.name,
                            'analysis_type': analysis_type,
                            'analysis': response.get('response', 'No analysis available') if response else 'No analysis available',
                            'timestamp': datetime.now().isoformat()
                        }

                        st.success("✅ Document analysis completed!")

                    else:
                        st.error("❌ Could not extract content from the uploaded file.")

                except Exception as e:
                    st.error(f"❌ Error analyzing document: {str(e)}")

    # Display previous analysis if available
    if 'document_analysis' in st.session_state:
        analysis = st.session_state.document_analysis

        st.markdown("#### 📋 Analysis Results")

        # Analysis header
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 **File:** {analysis['file_name']}")
        with col2:
            st.info(f"🔍 **Type:** {analysis['analysis_type']}")
        with col3:
            st.info(f"🕒 **Analyzed:** {datetime.fromisoformat(analysis['timestamp']).strftime('%Y-%m-%d %H:%M')}")

        # Analysis content
        st.markdown("##### 📊 AI Analysis Report")
        st.write(analysis['analysis'])

        # Follow-up questions section
        st.markdown("#### 💬 Follow-up Analysis")
        st.markdown("Ask follow-up questions about this analysis:")

        followup_question = st.text_input(
            "Follow-up Question",
            placeholder="e.g., What are the key risks in this forecast?",
            key="followup_question"
        )

        if st.button("🔍 Analyze Question", key="followup_button"):
            if followup_question.strip():
                with st.spinner("🤖 Analyzing follow-up question..."):
                    followup_prompt = f"""
                    Based on the previous analysis of the document "{analysis['file_name']}", please answer this follow-up question:

                    FOLLOW-UP QUESTION: {followup_question}

                    PREVIOUS ANALYSIS CONTEXT:
                    {analysis['analysis']}

                    Provide a focused, specific answer to this question.
                    """

                    followup_response = st.session_state.chatbot.chat(followup_prompt) if st.session_state.get('chatbot') else None

                    st.markdown("##### 🤖 AI Response")
                    if followup_response and followup_response.get('success'):
                        st.info(followup_response.get('response', 'No response'))
                    else:
                        st.error(f"Error: {followup_response.get('error', 'Unknown error') if followup_response else 'No response'}")


def main_chatbot_page():
    """Main chatbot interface with split screen design"""
    st.markdown("""
    <div class="sidebar-header">
        <h2>🤖 AI Retail Assistant</h2>
        <p>Your intelligent forecasting companion</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chatbot
    if not initialize_chatbot():
        # continue — initialization shows warnings; core features still usable
        pass

    # Create tabs for split screen functionality
    tab1, tab2 = st.tabs(["💬 AI Retail Assistant", "📊 AI Document Analyzer"])

    with tab1:
        st.markdown("### 💬 AI Retail Assistant")
        st.markdown("Interactive chatbot for real-time retail forecasting assistance")

        # Main chat area
        render_chat_history()

        # Handle input and interactions
        chat_input, send_button = render_input_area()

        # Handle chat input
        handle_chat_input(chat_input, send_button)

        # Handle image upload
        handle_image_upload()

        # Auto-scroll to bottom
        render_auto_scroll()

    with tab2:
        # AI Document Analyzer section
        render_ai_analyzer()

    # Sidebar
    render_global_sidebar()
    with st.sidebar:
        render_local_actions()


if __name__ == "__main__":
    main_chatbot_page()
