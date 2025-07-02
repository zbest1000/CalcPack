# CalcPack

> Modular calculator suite for engineering formulas with integrated PaddleOCR capabilities

## Overview

CalcPack is a modular calculator suite designed for engineering formulas and modern engineering workflows. This project now includes comprehensive PaddleOCR integration, providing state-of-the-art OCR capabilities including text detection, recognition, document structure analysis, and intelligent document understanding.

## Features

### Core Features
- Professional engineering tool
- Modern tech stack implementation
- Production-ready architecture
- Comprehensive documentation

### PaddleOCR Integration
- **PP-OCRv5**: High-accuracy text detection and recognition
- **PP-StructureV3**: Document structure analysis with layout detection, table recognition, and formula extraction
- **PP-ChatOCRv4**: Intelligent document understanding with key information extraction
- **MCP Server**: Model Context Protocol server for seamless integration with AI applications
- **Multi-language Support**: 80+ languages including Chinese, English, French, German, Korean, Japanese
- **Multiple Deployment Options**: Docker, standalone service, CLI, and API

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- Git

### Installation

#### Option 1: Direct Installation
```bash
git clone https://github.com/zbest1000/CalcPack.git
cd CalcPack

# Install dependencies
pip install -r requirements.txt

# Start the PaddleOCR service
python paddleocr_service.py serve
```

#### Option 2: Docker Deployment
```bash
git clone https://github.com/zbest1000/CalcPack.git
cd CalcPack

# Build and start services
docker-compose up -d
```

### Basic Usage

#### 1. OCR via API
```bash
# Text detection and recognition
curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image_data",
    "lang": "ch",
    "use_doc_orientation_classify": false
  }'
```

#### 2. Document Structure Analysis
```bash
# Structure analysis with PP-StructureV3
curl -X POST "http://localhost:8000/structure" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image_data",
    "output_format": "json",
    "use_table_recognition": true
  }'
```

#### 3. CLI Usage
```bash
# Process image with OCR
python paddleocr_service.py ocr image.jpg --lang ch --output result.json

# Start MCP server
python mcp_server.py --service-url http://localhost:8000
```

#### 4. Python API
```python
from paddleocr_service import PaddleOCRService
import asyncio

# Initialize service
service = PaddleOCRService()

# OCR request
request = OCRRequest(
    image_data="base64_encoded_image",
    lang="ch"
)

# Process
result = asyncio.run(service.process_ocr(request))
print(result)
```

## API Endpoints

### Core OCR Endpoints
- `POST /ocr` - Complete OCR (detection + recognition)
- `POST /text_detection` - Text detection only
- `POST /text_recognition` - Text recognition only
- `POST /structure` - Document structure analysis
- `POST /chat_ocr` - Intelligent document understanding
- `POST /upload_ocr` - Upload file for OCR processing
- `GET /health` - Service health check

### MCP Server Tools
- `ocr_text_detection_recognition` - Complete OCR processing
- `structure_analysis` - Document structure analysis
- `text_detection_only` - Text detection
- `text_recognition_only` - Text recognition
- `chat_ocr` - Key information extraction

## Configuration

### Service Configuration
Edit `config.yaml` to customize:
- Service settings (host, port, workers)
- OCR model configurations
- Performance settings
- Security options
- Storage locations

### Environment Variables
```bash
# Service settings
PADDLE_OCR_LOG_LEVEL=INFO
PADDLEOCR_SERVICE_URL=http://localhost:8000

# Performance
PADDLE_OCR_CPU_THREADS=4
PADDLE_OCR_ENABLE_GPU=false
```

## Supported Features

### OCR Capabilities
- **Text Detection**: Accurate text region detection
- **Text Recognition**: Multi-language text recognition
- **Document Orientation**: Automatic orientation correction
- **Text Line Orientation**: Individual text line orientation classification
- **Handwriting Recognition**: Support for handwritten text

### Document Analysis
- **Layout Detection**: Document structure analysis
- **Table Recognition**: Extract tables with complex structures
- **Formula Recognition**: Mathematical formula extraction
- **Seal Recognition**: Chinese seal/stamp recognition
- **Chart Recognition**: Chart and graph analysis

### Supported Languages
Chinese (Simplified/Traditional), English, French, German, Korean, Japanese, Italian, Spanish, Portuguese, Russian, Arabic, and 70+ more languages.

## Development

### Project Structure
```
CalcPack/
├── paddleocr_service.py    # Main service implementation
├── mcp_server.py           # MCP server implementation
├── requirements.txt        # Python dependencies
├── config.yaml            # Service configuration
├── docker-compose.yml     # Docker deployment
├── Dockerfile             # Service container
├── Dockerfile.mcp         # MCP server container
└── README.md              # This file
```

### Development Phases
- [x] Phase 1: Core PaddleOCR integration
- [x] Phase 2: MCP server implementation
- [ ] Phase 3: Advanced features and optimizations

### Tech Stack
- **Backend**: Python 3.9+, FastAPI, PaddleOCR 3.1.0
- **Models**: PP-OCRv5, PP-StructureV3, PP-ChatOCRv4
- **Deployment**: Docker, Docker Compose
- **API**: REST API, MCP (Model Context Protocol)
- **Monitoring**: Health checks, metrics

## Performance Optimization

### Model Options
- **Mobile Models**: Lightweight, faster inference
- **Server Models**: Higher accuracy, more resources
- **Custom Models**: Fine-tuned for specific use cases

### Hardware Acceleration
- CPU optimization with MKL-DNN
- GPU support (CUDA, ROCm)
- Chinese AI accelerators (Kunlun, Ascend)

## Deployment Options

### 1. Local Development
```bash
python paddleocr_service.py serve --host 0.0.0.0 --port 8000
```

### 2. Docker Container
```bash
docker build -t paddleocr-service .
docker run -p 8000:8000 paddleocr-service
```

### 3. Docker Compose
```bash
docker-compose up -d
```

### 4. Kubernetes
```bash
# Apply Kubernetes manifests (coming soon)
kubectl apply -f k8s/
```

## Monitoring and Logging

### Health Checks
- Service health endpoint: `GET /health`
- Docker health checks included
- Kubernetes readiness/liveness probes

### Metrics
- Request/response tracking
- Performance metrics
- Error rate monitoring
- Resource usage statistics

## Security

### API Security
- CORS configuration
- Rate limiting (optional)
- Input validation
- Error handling

### Data Security
- No persistent storage of images
- Temporary file cleanup
- Secure API key handling

## Troubleshooting

### Common Issues

1. **Service startup fails**
   ```bash
   # Check logs
   docker-compose logs paddleocr-service
   
   # Verify dependencies
   pip install -r requirements.txt
   ```

2. **Out of memory errors**
   ```bash
   # Use mobile models
   # Reduce batch size
   # Enable memory optimization
   ```

3. **Model download issues**
   ```bash
   # Check network connectivity
   # Verify model URLs
   # Use local model files
   ```

### Performance Tuning
- Adjust CPU threads in configuration
- Use GPU acceleration if available
- Optimize batch processing
- Enable MKL-DNN acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/zbest1000/CalcPack.git
cd CalcPack

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
python paddleocr_service.py serve --reload
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Excellent OCR toolkit
- [PaddlePaddle](https://www.paddlepaddle.org.cn/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - AI integration standard

## Support

- GitHub Issues: [Report bugs and feature requests](https://github.com/zbest1000/CalcPack/issues)
- Documentation: [Comprehensive guides](docs/)
- Examples: [Usage examples](examples/)

---

For more detailed information, please refer to the documentation in the `docs/` directory.
