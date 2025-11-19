#!/bin/bash

# AegisIsle setup script for local development
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Setting up AegisIsle development environment..."

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1,2)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        print_error "Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Docker is recommended for full functionality."
        print_warning "You can still run the application locally without Docker."
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed."
    fi

    print_status "Requirements check completed."
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate

    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip

    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt

    print_status "Python environment setup completed."
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration files..."

    # Copy environment file
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your API keys and configuration."
    else
        print_status ".env file already exists."
    fi

    print_status "Configuration setup completed."
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."

    mkdir -p data/uploads
    mkdir -p data/processed
    mkdir -p logs
    mkdir -p models/local
    mkdir -p notebooks

    print_status "Directories created."
}

# Setup pre-commit hooks
setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."

    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_status "Pre-commit hooks installed."
    else
        print_warning "pre-commit not found. Installing..."
        pip install pre-commit
        pre-commit install
        print_status "Pre-commit hooks installed."
    fi
}

# Run initial tests
run_tests() {
    print_status "Running initial tests..."

    # Run basic import test
    python3 -c "from src.aegis_isle.core.config import settings; print('✓ Configuration loaded successfully')"
    python3 -c "from src.aegis_isle.core.logging import logger; logger.info('✓ Logging configured successfully')"

    # Run pytest if available
    if command -v pytest &> /dev/null; then
        print_status "Running pytest..."
        pytest tests/ -v || print_warning "Some tests failed. This is normal for initial setup."
    else
        print_warning "pytest not found. Install it with: pip install pytest"
    fi

    print_status "Initial tests completed."
}

# Main setup function
main() {
    print_status "Starting AegisIsle development setup..."

    # Run setup steps
    check_requirements
    setup_directories
    setup_config
    setup_python_env
    setup_pre_commit
    run_tests

    print_status "Setup completed successfully!"

    echo ""
    print_status "Next steps:"
    echo "1. Edit .env file with your API keys and configuration"
    echo "2. Start development server:"
    echo "   - Local: source venv/bin/activate && uvicorn src.aegis_isle.api.main:app --reload"
    echo "   - Docker: ./scripts/deploy.sh --env development"
    echo "3. Visit http://localhost:8000/docs for API documentation"
    echo ""
    print_status "Happy coding! 🚀"
}

# Run main function
main "$@"