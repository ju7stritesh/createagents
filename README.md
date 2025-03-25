# CreateAgents

A Flask-based web application that enables intelligent document interaction using vector embeddings and LlamaParse. The application allows users to upload documents and create a conversational flow to interact with them using AI-powered chat.

## Features

- Document processing using LlamaParse for accurate text extraction
- Vector embeddings for efficient document retrieval and context management
- AI-powered chat interface for document queries
- Support for multiple file formats (PDF, DOCX, XLSX, CSV)
- Web content processing and integration
- Source attribution for AI responses
- Document management and organization

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini AI
- LlamaParse API Key for document processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd createagents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
LLAMA_PARSE_KEY=your_llama_parse_key
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5006`

3. Upload a document using the file upload interface

4. Use the chat interface to interact with your documents

## Project Structure

```
createagents/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
├── README.md          # Project documentation
└── templates/
    └── index.html     # Main template
```

## API Endpoints

- `GET /`: Main page
- `POST /upload`: Upload a document
- `POST /chat`: Send a question about the document
- `POST /delete_document`: Delete a document
- `GET /list_documents`: List all uploaded documents
- `POST /clear_chat`: Clear chat history
- `POST /process_url`: Process web content from a URL
- `GET /uploads/<filename>`: Serve uploaded files

## Dependencies

- Flask: Web framework
- LangChain: Document processing and AI integration
- Google Generative AI: AI model for document understanding
- LlamaParse: Advanced document parsing service
- FAISS: Vector similarity search
- PyMuPDF: PDF processing
- python-docx: DOCX file processing
- pandas: Data processing
- Pillow: Image processing

## License

This project is licensed under the MIT License - see the LICENSE file for details. 