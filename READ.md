# 🎓 AI Research Paper Generator

An intelligent web application that generates IEEE-format research papers using AI, with features like OCR, plagiarism detection, and automatic figure generation.

## Features

- ⚡ **Fast Generation**: Creates 4+ page papers in 30-45 seconds using parallel processing
- 📄 **IEEE Format**: Authentic two-column IEEE-style PDF and DOCX export
- 📊 **Figures & Tables**: Auto-generates bar charts, line graphs, and data tables
- 🔍 **Plagiarism Detection**: Multi-factor analysis for text uniqueness
- 🖼️ **OCR Support**: Extract topics from images using EasyOCR
- 🔗 **DOI Generation**: Realistic DOI assignment for papers
- 📚 **Citation Management**: Auto-generated references in IEEE format

## Tech Stack

- **Backend**: Python Flask
- **AI Model**: Ollama (Mistral)
- **OCR**: EasyOCR
- **PDF Generation**: ReportLab
- **Visualization**: Matplotlib
- **Frontend**: HTML, CSS, JavaScript

## Installation

### Prerequisites

- Python 3.8+
- Ollama installed ([Download](https://ollama.ai))
- Git

### Setup

1. Clone the repository:

git clone https://github.com/AbhishekJacobSanthosh/ai-paper-generator.git
cd ai-paper-generator

text

2. Create virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

3. Install dependencies:

pip install -r requirements.txt

text

4. Download Ollama model:

ollama pull mistral

text

## Usage

1. **Start Ollama** (in one terminal):

ollama run mistral

text

2. **Run the application** (in another terminal):

python app.py

text

3. **Open browser**:

http://localhost:8080

text

4. Enter your research topic, author details, and click **"Generate Research Paper"**!

## Project Structure

ai-paper-generator/
├── app.py # Main Flask application
├── templates/
│ └── index.html # Frontend UI
├── uploads/ # Temporary OCR uploads (gitignored)
├── saved_papers/ # Saved papers (gitignored)
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

text

## How It Works

1. **Input**: User enters research topic and author information
2. **AI Generation**: Ollama/Mistral generates 7 sections in parallel:
   - Abstract, Introduction, Literature Review
   - Methodology, Results, Discussion, Conclusion
3. **Visualization**: Matplotlib creates bar charts and line graphs
4. **Export**: ReportLab generates IEEE-format PDF with two-column layout
5. **Plagiarism Check**: Analyzes text uniqueness using multi-factor algorithm

## Features in Detail

### Paper Sections Generated
- **Abstract** (180+ words)
- **Introduction** (350+ words)
- **Literature Review** (350+ words)
- **Methodology** (350+ words)
- **Results** (300+ words)
- **Discussion** (350+ words)
- **Conclusion** (250+ words)
- **References** (IEEE format)

### Figures & Tables
- **Figure 1**: Performance comparison bar chart
- **Figure 2**: Training progress line graph
- **Table 1**: Comparative metrics

### Export Formats
- IEEE two-column PDF
- Microsoft Word DOCX
- Plain text
- JSON

## API Endpoints

- `POST /api/generate-paper` - Generate research paper
- `POST /api/ocr-generate` - Extract text from image
- `POST /api/plagiarism-check` - Check paper uniqueness
- `POST /api/export-pdf` - Export to PDF
- `POST /api/export-docx` - Export to DOCX

## Configuration

Optional: Configure in `app.py`:

MODEL_NAME = "mistral" # Change AI model
OLLAMA_API_URL = "http://..." # Ollama endpoint
WINSTON_API_KEY = "your_key" # Optional: Real plagiarism API

text

## Troubleshooting

**Issue**: `Connection refused to Ollama`
- **Solution**: Make sure Ollama is running: `ollama run mistral`

**Issue**: `Module not found`
- **Solution**: Activate venv and install dependencies: `pip install -r requirements.txt`

**Issue**: `Port 8080 already in use`
- **Solution**: Change port in `app.py`: `app.run(port=8081)`

**Issue**: OCR not working properly
- **Solution**: Use typed/printed text images for best results

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

## Future Enhancements

- [ ] User authentication and saved papers
- [ ] Multiple paper formats (APA, MLA, Chicago)
- [ ] Real-time collaboration
- [ ] Template customization
- [ ] LaTeX export
- [ ] Cloud deployment

## Acknowledgments

- **Ollama** - Local AI infrastructure
- **EasyOCR** - OCR capabilities
- **ReportLab** - PDF generation
- **Flask** - Web framework

## Author

**Abhishek Jacob Santhosh**
- Department of Computer Science and Engineering
- M.S. Ramaiah Institute of Technology (MSRIT)
- Email: 1ms22cs006@msrit.edu
- GitHub: [@AbhishekJacobSanthosh](https://github.com/AbhishekJacobSanthosh)

---

## Support

⭐ Star this repository if you find it helpful!

For issues or questions:
- Open an issue on GitHub
- Email: 1ms22cs006@msrit.edu

---

**Built with ❤️ at MSRIT | © 2025**
