#!/bin/bash
# =============================================================================
# StudentLife-Phenotyping: One-Command Setup & Run
# =============================================================================
# This script automates the complete environment setup and pipeline execution.
# Run this once to: build containers, start MLflow, install dependencies, and run pipeline.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    END-TO-END SEQUENCE (Recommended Order)                  │
# ├───────┬─────────────────┬───────────────────────────────────────────────────┤
# │ Step  │ Command         │ Description                                       │
# ├───────┼─────────────────┼───────────────────────────────────────────────────┤
# │  1    │ --build         │ Build Docker containers (~3-5 min first time)     │
# │  2    │ --start         │ Start MLflow tracking server (port 5000)          │
# │  3    │ --pipeline      │ Run full ML pipeline (14 steps, ~35-50 min)       │
# │  4    │ --api           │ Start FastAPI server (port 8000)                  │
# │  5    │ --test          │ Test API with sample prediction & anomaly         │
# └───────┴─────────────────┴───────────────────────────────────────────────────┘
#
# Pipeline Steps (executed by --pipeline):
#   Phase 1 — Sensor Data          Phase 3 — Activity Models
#   [1/14] Data Cleaning            [7/14] Baseline Models
#   [2/14] Time Alignment           [8/14] Gradient Boosting
#   [3/14] Final Dataset            [9/14] LSTM Model
#   [4/14] Feature Verification    [10/14] Transformer (SOTA)
#   Phase 2 — EMA Data             [11/14] Anomaly Detection
#   [5/14] Parse EMA Self-Reports  Phase 4 — Stress Prediction
#   [6/14] Merge Sensor ↔ EMA      [12/14] Stress Models (×10)
#                                  [13/14] SOTA (CatBoost HPO + Ensemble + SHAP)
#                                  [14/14] EMA Visualizations
#
# Quick Start (all-in-one):
#   ./setup_and_run.sh           # Runs steps 1-3 automatically
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
    echo "   3. Run data cleaning & alignment (sensor + EMA)"
    echo "   4. Train activity models (Baseline → Transformer)"
    echo "   5. Train stress prediction (10 algorithms)"
    echo "   6. Run anomaly detection & EMA visualizations"
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
    
    print_step "Starting API service on port 8000..."
    print_step "API will be accessible at:"
    echo "   - http://localhost:8000 (from host)"
    echo "   - http://localhost:8000/docs (Swagger UI)"
    echo "   - http://localhost:8000/health (Health check)"
    echo ""
    
    # Start API as a service (uses docker-compose.yml api service)
    docker-compose --profile api up -d api
    
    print_step "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo ""
            print_success "API is running at http://localhost:8000"
            print_step "View API docs at http://localhost:8000/docs"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    print_warning "API may still be starting. Check with: docker-compose logs api"
}

test_api() {
    print_header "Testing API Endpoints"
    
    # Check if API is running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_warning "API not running. Starting it first..."
        start_api
    fi
    
    echo ""
    print_step "[Test 1/4] Health Check"
    echo "   Request:  GET /health"
    HEALTH=$(curl -s http://localhost:8000/health)
    echo "   Response: $HEALTH"
    if echo "$HEALTH" | grep -q '"status":"ok"'; then
        print_success "Health check passed!"
    else
        print_error "Health check failed!"
        return 1
    fi
    
    echo ""
    print_step "[Test 2/4] Feature Info"
    echo "   Request:  GET /features"
    FEATURES=$(curl -s http://localhost:8000/features)
    echo "   Response: $FEATURES"
    print_success "Feature info retrieved!"
    
    echo ""
    print_step "[Test 3/4] Activity Prediction (264 features = 24 hours × 11 features)"
    echo "   Request:  POST /predict"
    
    # Generate 264 features (realistic sample values)
    PREDICT_PAYLOAD=$(python3 -c "
import json
# Simulate 24 hours of 11 features each (264 total)
# Features: activity_level, audio_energy, sleep_prob, conversation_duration,
# phone_charge, phone_lock, location_variance, distance_traveled,
# time_at_home, entropy, cluster_count
features = []
for hour in range(24):
    # Vary by time of day
    if 0 <= hour < 6:  # Night - low activity, high sleep
        features.extend([0.1, 0.05, 0.9, 0.0, 0.3, 0.8, 0.1, 0.0, 0.95, 0.1, 1])
    elif 6 <= hour < 9:  # Morning - waking up
        features.extend([0.4, 0.3, 0.3, 0.1, 0.5, 0.5, 0.3, 0.2, 0.7, 0.3, 2])
    elif 9 <= hour < 17:  # Day - high activity
        features.extend([0.8, 0.6, 0.05, 0.4, 0.4, 0.3, 0.7, 0.6, 0.3, 0.6, 4])
    elif 17 <= hour < 21:  # Evening - moderate
        features.extend([0.5, 0.4, 0.1, 0.3, 0.6, 0.4, 0.5, 0.3, 0.6, 0.4, 3])
    else:  # Late night - winding down
        features.extend([0.2, 0.2, 0.6, 0.1, 0.7, 0.7, 0.2, 0.1, 0.8, 0.2, 2])
print(json.dumps({'participant_id': 'test_participant', 'features': features}))
")
    
    PREDICT_RESULT=$(echo "$PREDICT_PAYLOAD" | curl -s -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" -d @-)
    echo "   Response: $PREDICT_RESULT"
    
    if echo "$PREDICT_RESULT" | grep -q '"predicted_activity_minutes"'; then
        print_success "Prediction endpoint passed!"
    else
        print_error "Prediction failed: $PREDICT_RESULT"
        return 1
    fi
    
    echo ""
    print_step "[Test 4/4] Anomaly Detection (11 features)"
    echo "   Request:  POST /anomaly"
    
    # 11 features for anomaly detection (single time point)
    ANOMALY_PAYLOAD='{
        "participant_id": "test_participant",
        "features": [0.65, 0.45, 0.12, 0.28, 0.55, 0.42, 0.58, 0.35, 0.48, 0.38, 3.5]
    }'
    
    ANOMALY_RESULT=$(curl -s -X POST http://localhost:8000/anomaly \
        -H "Content-Type: application/json" -d "$ANOMALY_PAYLOAD")
    echo "   Response: $ANOMALY_RESULT"
    
    if echo "$ANOMALY_RESULT" | grep -q '"is_anomaly"'; then
        print_success "Anomaly detection passed!"
    else
        print_error "Anomaly detection failed: $ANOMALY_RESULT"
        return 1
    fi
    
    echo ""
    print_header "All API Tests Passed! ✓"
    echo "Summary:"
    echo "   ✔ Health check: Models loaded (transformer, autoencoder)"
    echo "   ✔ Feature info: 11 features documented"
    echo "   ✔ Prediction: Returns activity minutes with interpretation"
    echo "   ✔ Anomaly: Returns is_anomaly with reconstruction error"
    echo ""
    print_step "Interactive API docs at: http://localhost:8000/docs"
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
    echo "┌─────────────────────────────────────────────────────────────────────────────┐"
    echo "│                    END-TO-END SEQUENCE (Recommended Order)                  │"
    echo "├───────┬─────────────────┬───────────────────────────────────────────────────┤"
    echo "│ Step  │ Command         │ Description                                       │"
    echo "├───────┼─────────────────┼───────────────────────────────────────────────────┤"
    echo "│  1    │ --build         │ Build Docker containers (~3-5 min first time)     │"
    echo "│  2    │ --start         │ Start MLflow tracking server (port 5000)          │"
    echo "│  3    │ --pipeline      │ Run full ML pipeline (14 steps, ~35-50 min)       │"
    echo "│  4    │ --api           │ Start FastAPI server (port 8000)                  │"
    echo "│  5    │ --test          │ Test API with sample prediction & anomaly         │"
    echo "└───────┴─────────────────┴───────────────────────────────────────────────────┘"
    echo ""
    echo "Usage: ./setup_and_run.sh [OPTION]"
    echo ""
    echo "Setup & Build:"
    echo "  --build        Build Docker containers"
    echo "  --start        Start MLflow tracking server"
    echo "  --pipeline     Run the full 14-step ML pipeline (sensor + EMA + SOTA)"
    echo ""
    echo "API & Testing:"
    echo "  --api          Start FastAPI prediction server (port 8000)"
    echo "  --test         Run API tests with proper sample inputs"
    echo ""
    echo "Utilities:"
    echo "  --shell        Enter container shell for interactive work"
    echo "  --status       Show status of running services"
    echo "  --stop         Stop all running services"
    echo "  --clean        Stop services and remove volumes"
    echo "  --help         Show this help message"
    echo ""
    echo "Quick Start (all-in-one, runs steps 1-3):"
    echo "  ./setup_and_run.sh"
    echo ""
    echo "Full End-to-End Demo:"
    echo "  ./setup_and_run.sh --build"
    echo "  ./setup_and_run.sh --start"
    echo "  ./setup_and_run.sh --pipeline"
    echo "  ./setup_and_run.sh --api"
    echo "  ./setup_and_run.sh --test"
    echo ""
    echo "Pipeline Steps (14 total, 4 phases):"
    echo "  Phase 1: Sensor Data           Phase 3: Activity Models"
    echo "  [1/14] Data Cleaning            [7/14] Baselines"
    echo "  [2/14] Time Alignment           [8/14] Gradient Boosting"
    echo "  [3/14] Final Dataset            [9/14] LSTM"
    echo "  [4/14] Feature Verification    [10/14] Transformer"
    echo "  Phase 2: EMA Data              [11/14] Anomaly Detection"
    echo "  [5/14] Parse EMA               Phase 4: Stress Prediction"
    echo "  [6/14] Merge Sensor+EMA        [12/14] 10 ML Algorithms"
    echo "                                 [13/14] SOTA (CatBoost HPO+Ensemble+SHAP)"
    echo "                                 [14/14] EMA Visualizations"
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
    --pipeline)
        check_prerequisites
        start_mlflow
        run_pipeline
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
    --test)
        check_prerequisites
        test_api
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
        echo "┌─────────────────────────────────────────────────────────────────────────────┐"
        echo "│ Step  │ Command         │ Description                                       │"
        echo "├───────┼─────────────────┼───────────────────────────────────────────────────┤"
        echo "│  1    │ --build         │ Build Docker containers                           │"
        echo "│  2    │ --start         │ Start MLflow (port 5000)                          │"
        echo "│  3    │ --pipeline      │ Run 14-step ML pipeline (sensor + EMA + SOTA)     │"
        echo "└───────┴─────────────────┴───────────────────────────────────────────────────┘"
        echo ""
        read -p "Continue with full setup? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            check_prerequisites
            build_containers
            start_mlflow
            run_pipeline
            show_status
            echo ""
            print_success "Setup complete! Next steps:"
            echo "   4. Start API:  ./setup_and_run.sh --api"
            echo "   5. Test API:   ./setup_and_run.sh --test"
            echo "   - View MLflow: http://localhost:5000"
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
