#!/bin/bash

# AegisIsle deployment script
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

# Default values
ENVIRONMENT="development"
COMPOSE_FILE="docker-compose.yml"
PULL_IMAGES=true
BUILD_IMAGES=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-pull)
            PULL_IMAGES=false
            shift
            ;;
        --no-build)
            BUILD_IMAGES=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env <environment>   Set environment (development|production) [default: development]"
            echo "  --no-pull            Skip pulling latest images"
            echo "  --no-build           Skip building images"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set compose file based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
elif [ "$ENVIRONMENT" = "development" ]; then
    COMPOSE_FILE="docker-compose.dev.yml"
fi

print_status "Deploying AegisIsle with environment: $ENVIRONMENT"
print_status "Using compose file: $COMPOSE_FILE"

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Compose file $COMPOSE_FILE not found!"
    exit 1
fi

# Check if .env file exists for production
if [ "$ENVIRONMENT" = "production" ] && [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Please edit .env file with your actual configuration before deployment!"
        exit 1
    else
        print_error ".env.example file not found!"
        exit 1
    fi
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/uploads data/processed logs models/local

# Set proper permissions
if [ "$ENVIRONMENT" = "production" ]; then
    print_status "Setting up production directories..."
    sudo mkdir -p /opt/aegis-isle/{data/{postgres,redis,qdrant,app},logs,models}
    sudo chown -R $USER:$USER /opt/aegis-isle/
fi

# Pull latest images if requested
if [ "$PULL_IMAGES" = true ]; then
    print_status "Pulling latest images..."
    docker-compose -f $COMPOSE_FILE pull
fi

# Build images if requested
if [ "$BUILD_IMAGES" = true ]; then
    print_status "Building images..."
    docker-compose -f $COMPOSE_FILE build
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down

# Start services
print_status "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Health check
print_status "Performing health check..."
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    print_status "Health check passed! AegisIsle is running."
else
    print_warning "Health check failed. Checking logs..."
    docker-compose -f $COMPOSE_FILE logs --tail=20 aegis-isle
fi

# Show running services
print_status "Running services:"
docker-compose -f $COMPOSE_FILE ps

print_status "Deployment complete!"

if [ "$ENVIRONMENT" = "development" ]; then
    echo ""
    print_status "Development URLs:"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Jupyter: http://localhost:8888"
    echo "  PGAdmin: http://localhost:5050"
    echo "  Qdrant: http://localhost:6334/dashboard"
elif [ "$ENVIRONMENT" = "production" ]; then
    echo ""
    print_status "Production deployment complete!"
    echo "  API: https://your-domain.com"
    echo "  Health: https://your-domain.com/api/v1/health"
fi