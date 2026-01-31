#!/bin/bash
# =============================================================================
# StudentLife-Phenotyping: One-Command Setup & Run
# =============================================================================
# This script automates the complete environment setup and pipeline execution.
# Run this once to: build containers, start MLflow, install dependencies, and run pipeline.
#
# Usage:
#   ./setup_and_run.sh           # Full setup + run pipeline
#   ./setup_and_run.sh --build   # Only build containers
#   ./setup_and_run.sh --start   # Only start services (skip build)
#   ./setup_and_run.sh --shell   # Enter container shell (interactive)
#   ./setup_and_run.sh --api     # Start API server
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${CYAN}$1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✖${NC} $1"
}

print_success() {
    echo -e "${GREEN}✔${NC} $1"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    print_success "Docker version: $DOCKER_VERSION"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is available"
    
    # Check disk space
    FREE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$FREE_SPACE" -lt 10 ]; then
        print_warning "Low disk space: ${FREE_SPACE}GB free. Recommend 20GB+."
    else
        print_success "Disk space: ${FREE_SPACE}GB available"
    fi
    
    # Create required directories
    mkdir -p models reports data notebooks .env.d
    
    # Check for .env file
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env from .env.example"
        else
            touch .env
            print_warning "Created empty .env file"
        fi
    fi
}

build_containers() {
    print_header "Building Docker Containers"
    print_step "This may take 3-5 minutes on first run..."
    
    docker-compose --profile training build
    
    print_success "Containers built successfully!"
}

start_mlflow() {
    print_header "Starting MLflow Tracking Server"
    
    # Check if MLflow is already running
    if docker-compose ps mlflow 2>/dev/null | grep -q "Up"; then
        print_success "MLflow is already running"
    else
        docker-compose up -d mlflow
        
        print_step "Waiting for MLflow to be healthy..."
        for i in {1..30}; do
            if docker-compose ps mlflow 2>/dev/null | grep -q "healthy"; then
                break
            fi
            sleep 2
            echo -n "."
        done
        echo ""
        
        if docker-compose ps mlflow 2>/dev/null | grep -q "healthy"; then
            print_success "MLflow is running at http://localhost:5000"
        else
            print_warning "MLflow may still be starting. Check with: docker-compose ps"
        fi
    fi
}

run_pipeline() {
    print_header "Running ML Pipeline in Container"
    
    print_step "Entering training container and running pipeline..."
    print_step "This will:"
    echo "   1. Check/download dataset"
    echo "   2. Install dependencies (if needed)"
    echo "   3. Run data cleaning & alignment"
    echo "   4. Train all models (Baseline → Transformer)"
    echo "   5. Run anomaly detection"
    echo ""
    
    docker-compose --profile training run --rm training bash -c "./run_pipeline.sh"
    
    print_success "Pipeline completed! Check MLflow at http://localhost:5000"
}

enter_shell() {
    print_header "Entering Container Shell"
    print_step "You can now run commands interactively."
    print_step "Type 'exit' to leave the container."
    echo ""
    
    docker-compose --profile training run --rm training bash
}

start_api() {
    print_header "Starting FastAPI Server"
    
    print_step "Starting API on port 8000..."
    print_step "API will be accessible at:"
    echo "   - http://localhost:8000 (from host)"
    echo "   - http://localhost:8000/docs (Swagger UI)"
    echo "   - http://localhost:8000/health (Health check)"
    echo ""
    print_step "Press Ctrl+C to stop the server."
    echo ""
    
    docker-compose --profile training run --rm -p 8000:8000 training \
        bash -c "source /app/.venv/bin/activate && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
}

show_status() {
    print_header "Service Status"
    
    docker-compose ps
    
    echo ""
    print_step "Useful URLs:"
    echo "   - MLflow UI:      http://localhost:5000"
    echo "   - API (if running): http://localhost:8000"
    echo "   - API Docs:       http://localhost:8000/docs"
}

show_help() {
    echo ""
    echo "StudentLife-Phenotyping Setup & Run Script"
    echo "==========================================="
    echo ""
    echo "Usage: ./setup_and_run.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  (no option)    Full setup: build + start MLflow + run pipeline"
    echo "  --build        Only build Docker containers"
    echo "  --start        Only start MLflow (skip build)"
    echo "  --shell        Enter container shell for interactive work"
    echo "  --api          Start the FastAPI prediction server"
    echo "  --status       Show status of running services"
    echo "  --stop         Stop all running services"
    echo "  --clean        Stop services and remove volumes"
    echo "  --help         Show this help message"
    echo ""
    echo "Quick Start (one command):"
    echo "  ./setup_and_run.sh"
    echo ""
    echo "Step-by-step (manual control):"
    echo "  1. ./setup_and_run.sh --build    # Build containers"
    echo "  2. ./setup_and_run.sh --start    # Start MLflow"
    echo "  3. ./setup_and_run.sh --shell    # Enter container"
    echo "  4. ./run_pipeline.sh             # Run pipeline (inside container)"
    echo "  5. ./setup_and_run.sh --api      # Start API server"
    echo ""
}

stop_services() {
    print_header "Stopping Services"
    docker-compose down
    print_success "All services stopped"
}

clean_all() {
    print_header "Cleaning Up (Removing Volumes)"
    docker-compose down -v
    print_success "Services stopped and volumes removed"
}

# =============================================================================
# Main Entry Point
# =============================================================================

case "${1:-}" in
    --build)
        check_prerequisites
        build_containers
        ;;
    --start)
        check_prerequisites
        start_mlflow
        show_status
        ;;
    --shell)
        check_prerequisites
        start_mlflow
        enter_shell
        ;;
    --api)
        check_prerequisites
        start_mlflow
        start_api
        ;;
    --status)
        show_status
        ;;
    --stop)
        stop_services
        ;;
    --clean)
        clean_all
        ;;
    --help|-h)
        show_help
        ;;
    "")
        # Full setup and run
        print_header "StudentLife-Phenotyping: Complete Setup"
        echo "This will build containers, start MLflow, and run the full ML pipeline."
        echo "Estimated time: 30-60 minutes (depending on hardware and network)"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            check_prerequisites
            build_containers
            start_mlflow
            run_pipeline
            show_status
            echo ""
            print_success "Setup complete! Next steps:"
            echo "   - View experiments: http://localhost:5000"
            echo "   - Start API: ./setup_and_run.sh --api"
            echo "   - Enter shell: ./setup_and_run.sh --shell"
        else
            echo "Aborted."
        fi
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
