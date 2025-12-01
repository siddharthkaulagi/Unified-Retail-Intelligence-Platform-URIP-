import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache

# Import our Gemini AI classes
try:
    from utils.gemini_ai import GeminiChatbot, GeminiAIAssistant
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Constants
MAX_CHAT_HISTORY = 50
MAX_IMAGE_SIZE = (800, 600)
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'gif', 'bmp']

def initialize_floating_chatbot() -> bool:
    """Initialize the floating chatbot if not already done"""
    if 'floating_chatbot' not in st.session_state:
        if not GEMINI_AVAILABLE:
            return False

        try:
            st.session_state.floating_chatbot = GeminiChatbot()
            st.session_state.floating_chat_history = []
            st.session_state.floating_chat_open = False
            st.session_state.floating_uploaded_image = None
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize floating chatbot: {str(e)}")
            return False
    return True

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
        return chat_history[-MAX_CHAT_HISTORY:]
    return chat_history

def render_floating_chat_button():
    """Render the floating chat button"""
    # Custom CSS for floating button and chat interface
    st.markdown("""
    <style>
        .floating-chat-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }
        .floating-chat-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        }
        .chat-interface {
            position: fixed;
            bottom: 100px;
            right: 20px;
            width: 380px;
            height: 500px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            z-index: 1001;
            display: none;
            flex-direction: column;
        }
        .chat-interface.open {
            display: flex;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            max-height: 320px;
        }
        .message {
            margin: 8px 0;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 85%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: #f1f3f4;
            color: black;
        }
        .chat-input-area {
            padding: 20px;
            border-top: 1px solid #eee;
        }
        .chat-input {
            width: 100%;
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 20px;
            outline: none;
            resize: none;
            font-family: inherit;
        }
        .chat-controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            align-items: center;
        }
        .send-button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 15px;
            cursor: pointer;
        }
        .image-upload-button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
        }
        .close-button {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-preview {
            max-width: 200px;
            max-height: 150px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>

    <!-- No inline toggle script; interactions use Streamlit session state via postMessage -->
    """, unsafe_allow_html=True)

    # Floating chat button
    st.markdown("""
    <button class="floating-chat-button" onclick="window.parent.postMessage({type:'streamlit:setSessionState', key:'floating_chat_open', value:true}, '*'); window.parent.postMessage({type:'streamlit:rerun'}, '*');" title="Open AI Chat">
        AI
    </button>
    """, unsafe_allow_html=True)

def render_floating_chat_interface():
    """Render the floating chat interface"""
    # Render chat interface server-side with 'open' class when session state says it's open
    is_open = st.session_state.get('floating_chat_open', False)
    open_class = ' chat-interface open' if is_open else ' chat-interface'
    st.markdown(f"""
    <div class="{open_class.strip()}">
        <div class="chat-header">
                <div>
                    <strong>AI Assistant</strong><br>
                    <small>Retail Forecasting Expert</small>
                </div>
                <button class="close-button" onclick="window.parent.postMessage({{type:'streamlit:setSessionState', key:'floating_chat_open', value:false}}, '*'); window.parent.postMessage({{type:'streamlit:rerun'}}, '*');">X</button>
            </div>

        <div class="chat-messages" id="chat-messages">
            <!-- Messages will be dynamically added here -->
        </div>

        <div class="chat-input-area">
            <textarea class="chat-input" id="chat-input" placeholder="Ask me anything about retail forecasting..." rows="2"></textarea>
            <div class="chat-controls">
                <input type="file" id="image-upload" accept="image/*" style="display: none;">
                <button class="image-upload-button" onclick="document.getElementById('image-upload').click()">
                    Image
                </button>
                <button class="send-button" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_floating_chat_scripts():
    """Render JavaScript for floating chat functionality with backend integration"""
    # This script uses Streamlit session state via postMessage. Guard DOM accesses so the script
    # won't throw if the chat interface is not present in the DOM.
    st.markdown("""
    <script>
        let chatHistory = [];
        let uploadedImage = null;

        function safeGet(id) {
            return document.getElementById(id);
        }

        function addMessage(message, isUser = true, image = null) {
            const messagesContainer = safeGet('chat-messages');
            if (!messagesContainer) return;
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

            let content = message;
            if (image) {
                content += `<br><img src="${image}" class="image-preview" style="margin-top: 10px;">`;
            }

            messageDiv.innerHTML = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showTyping() {
            addMessage('AI is typing...', false);
        }

        function hideTyping() {
            const typingIndicator = document.querySelector('.bot-message:last-child');
            if (typingIndicator && typingIndicator.textContent.includes('typing')) {
                typingIndicator.remove();
            }
        }

        function sendMessage() {
            const input = safeGet('chat-input');
            if (!input && !uploadedImage) return;
            const message = input ? input.value.trim() : '';

            if (!message && !uploadedImage) return;

            // Add user message to chat
            addMessage(message || '📷 Image uploaded', true, uploadedImage);

            // Clear input and image
            if (input) input.value = '';
            uploadedImage = null;

            // Show typing indicator
            showTyping();

            // Use Streamlit session state via postMessage
            window.parent.postMessage({type:'streamlit:setSessionState', key:'floating_chat_message', value: message}, '*');
            if (uploadedImage) {
                window.parent.postMessage({type:'streamlit:setSessionState', key:'floating_chat_image', value: uploadedImage}, '*');
            }
            window.parent.postMessage({type:'streamlit:rerun'}, '*');

            // After rerun, Streamlit will repopulate session state; we rely on loadChatHistory to show responses
            setTimeout(() => { loadChatHistory(); }, 600);
        }

        function loadChatHistory() {
            const messagesContainer = safeGet('chat-messages');
            if (!messagesContainer) return;

            // Clear existing messages
            while (messagesContainer.firstChild) {
                messagesContainer.removeChild(messagesContainer.firstChild);
            }

            // Placeholder bot message - real responses come from session state after rerun
            setTimeout(() => {
                addMessage('Thank you for your question! I\'m here to help with retail forecasting and data analysis.', false);
            }, 300);
        }

        // Handle image upload safely
        const imageUpload = safeGet('image-upload');
        if (imageUpload) {
            imageUpload.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        uploadedImage = e.target.result;
                        const input = safeGet('chat-input');
                        if (input) input.placeholder = 'Image uploaded! Ask me about it...';
                    };
                    reader.readAsDataURL(file);
                }
            });
        }

        const chatInput = safeGet('chat-input');
        if (chatInput) {
            chatInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        // Initialize chat history on load
        window.addEventListener('load', function() { loadChatHistory(); });
    </script>
    """, unsafe_allow_html=True)

def handle_floating_chat_message():
    """Handle floating chat messages using Streamlit session state"""
    if 'floating_chat_message' in st.session_state and st.session_state.floating_chat_message:
        message = st.session_state.floating_chat_message
        image = st.session_state.get('floating_chat_image')

        # Clear the message from session state
        st.session_state.floating_chat_message = None
        if 'floating_chat_image' in st.session_state:
            st.session_state.floating_chat_image = None

        # Get AI response
        if 'floating_chatbot' in st.session_state:
            try:
                if image:
                    response = st.session_state.floating_chatbot.analyze_image(image, message)
                else:
                    response = st.session_state.floating_chatbot.chat(message)

                if response and response.get('success', False):
                    bot_response = response.get('analysis', response.get('response', 'No response'))
                else:
                    bot_response = f"❌ Error: {response.get('error', 'Unknown error') if response else 'No response from AI'}"

                # Add to chat history
                if 'floating_chat_history' not in st.session_state:
                    st.session_state.floating_chat_history = []

                st.session_state.floating_chat_history.append({
                    'user': message,
                    'bot': bot_response,
                    'timestamp': datetime.now().isoformat(),
                    'image': image
                })

                # Keep only last 50 messages
                if len(st.session_state.floating_chat_history) > 50:
                    st.session_state.floating_chat_history = st.session_state.floating_chat_history[-50:]

                return bot_response
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.floating_chat_history.append({
                    'user': message,
                    'bot': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'image': image
                })
                return error_msg

    return None

def render_floating_chat_scripts():
    """Render JavaScript for floating chat functionality with Streamlit integration"""
    st.markdown("""
    <script>
        let chatHistory = [];
        let uploadedImage = null;

        function addMessage(message, isUser = true, image = null) {
            const messagesContainer = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

            let content = message;
            if (image) {
                content += `<br><img src="${image}" class="image-preview" style="margin-top: 10px;">`;
            }

            messageDiv.innerHTML = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showTyping() {
            addMessage('🤖 AI is typing...', false);
        }

        function hideTyping() {
            const typingIndicator = document.querySelector('.bot-message:last-child');
            if (typingIndicator && typingIndicator.textContent.includes('typing')) {
                typingIndicator.remove();
            }
        }

        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();

            if (!message && !uploadedImage) return;

            // Add user message to chat
            addMessage(message || '📷 Image uploaded', true, uploadedImage);

            // Clear input and image
            input.value = '';
            uploadedImage = null;
            input.placeholder = 'Ask me anything about retail forecasting...';

            // Show typing indicator
            showTyping();

            /* Use Streamlit's rerun mechanism to process the message
               Set session state variables that Streamlit will pick up */
            window.parent.postMessage({
                type: 'streamlit:setSessionState',
                key: 'floating_chat_message',
                value: message
            }, '*');

            if (uploadedImage) {
                window.parent.postMessage({
                    type: 'streamlit:setSessionState',
                    key: 'floating_chat_image',
                    value: uploadedImage
                }, '*');
            }

            // Trigger a Streamlit rerun
            window.parent.postMessage({
                type: 'streamlit:rerun'
            }, '*');

            // Wait a bit for Streamlit to process, then refresh the chat
            setTimeout(() => {
                loadChatHistory();
            }, 500);
        }

        function loadChatHistory() {
            // Get chat history from Streamlit session state
            // This is a simplified approach - in practice, you'd need a more sophisticated method
            const messagesContainer = document.getElementById('chat-messages');

            // Clear existing messages (except the current user message)
            while (messagesContainer.children.length > 1) {
                messagesContainer.removeChild(messagesContainer.lastChild);
            }

            // Remove typing indicator if present
            hideTyping();

            // Add a new bot message (this would be populated from session state)
            // In a real implementation, you'd fetch this from the server
            setTimeout(() => {
                addMessage('Thank you for your question! I\\'m here to help with retail forecasting and data analysis.', false);
            }, 500);
        }

        // Handle image upload
        document.getElementById('image-upload').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImage = e.target.result;
                    document.getElementById('chat-input').placeholder = 'Image uploaded! Ask me about it...';
                };
                reader.readAsDataURL(file);
            }
        });

        // Handle Enter key in textarea
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Initialize chat history
        window.onload = function() {
            // Load existing chat history
            chatHistory = [];
        };
    </script>
    """, unsafe_allow_html=True)

def display_floating_chat():
    """Main function to display the floating chat interface"""
    # Initialize chatbot
    if not initialize_floating_chatbot():
        return

    # Handle any incoming chat messages
    handle_floating_chat_message()

    # Render the floating chat components
    render_floating_chat_button()
    render_floating_chat_interface()
    render_floating_chat_scripts()

# Export the main function
__all__ = ['display_floating_chat']
