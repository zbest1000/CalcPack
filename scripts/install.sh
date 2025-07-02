#!/bin/bash

# PaddleOCR Integration Installation Script
# This script installs and configures the PaddleOCR service

set -e

echo "🚀 Starting PaddleOCR Integration Installation..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.9"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
        else
            print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9 or higher."
        exit 1
    fi
}

# Check system dependencies
check_system_deps() {
    print_status "Checking system dependencies..."
    
    # Check for required system packages
    MISSING_DEPS=()
    
    # Check for build essentials
    if ! command -v gcc &> /dev/null; then
        MISSING_DEPS+=("build-essential")
    fi
    
    # Check for curl
    if ! command -v curl &> /dev/null; then
        MISSING_DEPS+=("curl")
    fi
    
    # Check for wget
    if ! command -v wget &> /dev/null; then
        MISSING_DEPS+=("wget")
    fi
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warning "Missing system dependencies: ${MISSING_DEPS[*]}"
        print_status "Installing missing dependencies..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y "${MISSING_DEPS[@]}"
        elif command -v yum &> /dev/null; then
            sudo yum install -y "${MISSING_DEPS[@]}"
        elif command -v brew &> /dev/null; then
            brew install "${MISSING_DEPS[@]}"
        else
            print_error "Package manager not found. Please install dependencies manually: ${MISSING_DEPS[*]}"
            exit 1
        fi
    fi
    
    print_success "System dependencies checked"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        python3 -m pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p output
    mkdir -p models
    mkdir -p temp
    mkdir -p logs
    
    print_success "Directories created"
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Environment file created from .env.example"
            print_warning "Please edit .env file to configure your settings"
        else
            print_warning ".env.example not found, creating basic .env file"
            cat > .env << EOF
PADDLE_OCR_HOST=0.0.0.0
PADDLE_OCR_PORT=8000
PADDLE_OCR_LOG_LEVEL=INFO
PADDLE_OCR_DEFAULT_LANG=ch
EOF
            print_success "Basic .env file created"
        fi
    else
        print_success "Environment file already exists"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python3 -c "
import sys
try:
    import paddleocr
    print('✓ PaddleOCR import successful')
except ImportError as e:
    print(f'✗ PaddleOCR import failed: {e}')
    sys.exit(1)

try:
    import fastapi
    print('✓ FastAPI import successful')
except ImportError as e:
    print(f'✗ FastAPI import failed: {e}')
    sys.exit(1)

try:
    import cv2
    print('✓ OpenCV import successful')
except ImportError as e:
    print(f'✗ OpenCV import failed: {e}')
    sys.exit(1)

print('✓ All core dependencies imported successfully')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Download models (optional)
download_models() {
    print_status "Checking for pre-trained models..."
    
    read -p "Do you want to download pre-trained models now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Downloading models (this may take a while)..."
        
        # This will trigger model download on first run
        python3 -c "
from paddleocr import PaddleOCR
print('Initializing PaddleOCR (this will download models)...')
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
print('Model download completed!')
" || print_warning "Model download failed - models will be downloaded on first use"
        
        print_success "Model download completed"
    else
        print_status "Skipping model download - models will be downloaded on first use"
    fi
}

# Check Docker installation (optional)
check_docker() {
    print_status "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        
        if command -v docker-compose &> /dev/null; then
            print_success "Docker Compose found"
            
            read -p "Do you want to build Docker images? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_status "Building Docker images..."
                docker-compose build
                print_success "Docker images built successfully"
            fi
        else
            print_warning "Docker Compose not found - Docker deployment will not be available"
        fi
    else
        print_warning "Docker not found - Docker deployment will not be available"
    fi
}

# Main installation process
main() {
    echo "🔧 PaddleOCR Integration Setup"
    echo "================================"
    echo
    
    check_python
    check_system_deps
    install_python_deps
    create_directories
    setup_environment
    test_installation
    download_models
    check_docker
    
    echo
    echo "🎉 Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Edit .env file to configure your settings"
    echo "2. Start the service: python3 paddleocr_service.py serve"
    echo "3. Or use Docker: docker-compose up -d"
    echo "4. Access the API at: http://localhost:8000"
    echo "5. View API docs at: http://localhost:8000/docs"
    echo
    echo "For more information, see README.md"
}

# Run main function
main "$@"