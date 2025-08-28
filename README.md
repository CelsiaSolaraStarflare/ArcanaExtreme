![MIT License](https://img.shields.io/badge/license-CC.BY.ND.SA-green.svg)
![ArcanaLTE Version](https://img.shields.io/badge/version-ArcanaExtreme%201.1Beta9-black)
![Development Status](https://img.shields.io/badge/status-Beta-orange)
![Built with Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b?logo=streamlit)
![Powered by StandardCAS](https://img.shields.io/badge/built%20by-StandardCAS™-purple)

---

# 🌌 Welcome to Arcana Extreme  
✨ By StandardCAS™ Neon Group | ArcanaExtreme Standalone Version
[*Version 1.1 Beta 09 (2025-08-28)*]

--- 
## 📚 Table of Contents
1. [Introduction](https://github.com/CelsiaSolaraStarflare/Arcana/blob/ArcanaLTE/README.md#-introduction)
2. [Versions](https://github.com/CelsiaSolaraStarflare/Arcana/blob/ArcanaLTE/README.md#-available-versions)
3. [Make It Yours – Private & Custom Chatbots](https://github.com/CelsiaSolaraStarflare/Arcana/blob/ArcanaLTE/README.md#-make-it-private-and-customized)
4. [How to Use](https://github.com/CelsiaSolaraStarflare/Arcana/blob/ArcanaLTE/README.md#-how-to-use)
5. [Works Cited](https://github.com/CelsiaSolaraStarflare/Arcana/blob/ArcanaLTE/README.md#-works-cited)
6. [License](https://github.com/CelsiaSolaraStarflare/Arcana/blob/ArcanaLTE/README.md#-license)

---

## 💖 Introduction

Welcome to **Arcana Extreme** — your intelligent, dynamic study resource hub! Built to empower students with fast and accurate support, Arcana is designed to help you succeed in your learning journey through smart question answering, document indexing, and rich content generation.

---

## 🛡 Make It Private and Customized!

Want to keep your chatbot private and unique to your needs? We've got you covered. Arcana is built with **Streamlit**, which makes it easy to:

- Fork this repository 💻  
- Deploy your own **private chatbot server** 🔐  
- Customize settings, style, and plugins to your personal preferences 🎨

---

## 📚 Works Cited and Credits

```
For this Chatbot:
ArcanaExtreme, StandardCAS™　Neon Group. August 2025.
FiberDB, StandardCAS™. Chengjui Osmond Fan, Juilyn Celsia. April 3, 2025.

For the Development Team:
Chengjui Osmond Fan - Base algorithm coding, DBMS designing, API functions, and Accessories.

For the Utilized Technologies:
Qwen, Alibaba Cloud Group.
Streamlit 1.22, Streamlit.io.
```

---

Arcana Extreme is a sophisticated Streamlit application that transforms your documents into an intelligent, searchable knowledge base. With cutting-edge AI integration and beautiful visual interfaces, Arcana brings the power of artificial intelligence directly to your fingertips.

## ✨ Features

### 🧠 **Smart Document Processing**
- **Multi-format Support**: PDF, DOCX, PPTX, TXT, CSV files
- **Intelligent Indexing**: Advanced NLP-powered document analysis
- **Contextual Search**: Find exactly what you need with semantic understanding

### 💬 **AI-Powered Chat Interface**
- **Context-Aware Responses**: Chat with your documents using OpenAI GPT
- **Document Integration**: Automatic context injection from your knowledge base
- **Smart Conversations**: Maintains conversation history and context

### 🎨 **Visual Presentation Tools**
- **AI Presentation Generator**: Create stunning PowerPoint presentations automatically
- **Smart Content Creation**: Generate outlines, slides, and formatted content
- **Visual Slide Editor**: PowerPoint/Keynote-style editor with streamlit-theta
- **Document Export**: Export to both PowerPoint and Word formats

### 📚 **Study Guide Generator** *(NEW!)*
- **Multiple Study Styles**: Comprehensive, summary, outline, flashcard-prep, and exam-focused modes
- **Intelligent Outline Generation**: AI creates structured study guide outlines from your documents
- **Section-by-Section Content**: Detailed explanations, examples, and practice questions
- **Formatted Export**: Professional Word documents with table of contents and formatting
- **Cross-Platform Integration**: Seamlessly switch between study guides and presentations

### 🃏 **Q&A Flashcard Generator** *(NEW!)*
- **Quick Flashcard Creation**: Generate study flashcards from any topic
- **Interactive Review**: Expandable question/answer format for effective studying
- **Bulk Generation**: Create 5-20 flashcards at once
- **Export Options**: Download as text files for external flashcard apps

### 🎯 **Advanced Analytics**
- **Long-form Analysis**: Deep document analysis with Qwen models
- **Multiple Analysis Modes**: Summary, takeaways, pros/cons, simplified explanations
- **Rich Text Editing**: AI-powered document editor with diff visualization

### 🎙️ **Speech Integration**
- **Speech-to-Text**: Convert voice recordings to text using DashScope
- **Real-time Transcription**: Live voice input processing
- **Audio File Support**: Process uploaded audio files

### 🎨 **Beautiful Interface**
- **Modern UI**: Clean, intuitive design with dark/light themes
- **Responsive Layout**: Works seamlessly across different screen sizes
- **Icon-based Navigation**: Easy-to-use app-style interface

## 🚀 **NEW: Streamlit Theta v1.0.0**

We've built our own PowerPoint/Keynote-style visual editor for Streamlit! 

### Features:
- 🎨 Drag-and-drop text boxes, images, and shapes
- 📝 Real-time text editing with formatting controls
- 🖼️ Visual slide thumbnails and navigation
- 🎯 Property panels for precise styling
- 💾 Save and export functionality

```python
from streamlit_theta import theta_slide_editor

# Create a visual slide editor
slides = theta_slide_editor(
    slides=[...],  # Your slide data
    width=800,
    height=600,
    key="my_editor"
)
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- OpenAI API key
- DashScope API key (optional, for image generation and speech features)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/arcana/arcana-extreme.git
   cd arcana-extreme
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install streamlit-theta  # Our custom visual editor
   ```

3. **Set up environment variables**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   DASHSCOPE_API_KEY=your_dashscope_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run ArcanaExtreme.py
   ```

## 📚 Usage

### Document Management
1. **Upload Documents**: Use the Files page to upload and index your documents
2. **Smart Indexing**: Arcana automatically processes and indexes your content
3. **Search & Discovery**: Use the chat interface to ask questions about your documents

### AI Chat
1. **Start Conversations**: Ask questions in natural language
2. **Context Integration**: Arcana automatically includes relevant document context
3. **Rich Responses**: Get detailed, contextual answers with sources

### Presentation Creation
1. **Choose Topic**: Enter your presentation topic
2. **AI Generation**: Let AI create an outline and content
3. **Visual Editing**: Use our Theta editor to customize slides visually
4. **Export**: Download as PowerPoint or Word document

### Study Guide Creation *(NEW!)*
1. **Select Topic & Style**: Choose your study topic and preferred style (comprehensive, summary, outline, flashcard-prep, or exam-focused)
2. **Review Outline**: AI generates a structured outline that you can edit
3. **Content Generation**: AI creates detailed content for each section based on your documents
4. **Edit & Customize**: Review and modify content in tabbed interface
5. **Export Options**: Download as formatted Word document or plain text

### Flashcard Generation *(NEW!)*
1. **Enter Topic**: Specify what you want to create flashcards about
2. **Set Quantity**: Choose 5-20 flashcards
3. **AI Generation**: AI creates question-answer pairs
4. **Interactive Review**: Study with expandable Q&A format
5. **Export**: Download as text file for use in flashcard apps

## 🏗️ Architecture

Arcana Extreme is built with:

- **Frontend**: Streamlit with custom components
- **AI Engine**: OpenAI GPT models with DashScope integration
- **Document Processing**: Advanced NLP with NLTK and custom parsers
- **Search Engine**: FiberDBMS for fast, contextual document retrieval
- **Visual Editor**: Custom streamlit-theta component

## 📝 API Reference

### Core Components

#### FiberDBMS
```python
from fiber import FiberDBMS

dbms = FiberDBMS()
dbms.add_document("doc.pdf", content, keywords)
results = dbms.query("search terms", top_n=5)
```

#### Presentation Generator
```python
from mixup import create_presentation_from_content

ppt_path = create_presentation_from_content(
    presentation_content, 
    topic, 
    api_key
)
```

#### Theta Visual Editor
```python
from streamlit_theta import theta_slide_editor

slides = theta_slide_editor(
    slides=slide_data,
    width=800,
    height=600,
    key="editor"
)
```

*Transform your documents into intelligent conversations with Arcana Extreme!*


