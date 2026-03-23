#!/bin/bash

# Update README.md
sed -i 's/## Overview/## Overview\n\n### 🏗️ Architecture\n\n```mermaid\ngraph TD\n    A[Raw Sensor Data] --> B{Data Cleaning \& Alignment}\n    C[EMA Self-Reports] --> B\n    B --> D[Feature Engineering]\n    D --> E[Model Training \& Tuning]\n    E --> F[Ensemble \& Evaluation]\n    F --> G[FastAPI Service]\n    G --> H[End User]\n```\n/' README.md

# Update API section info for non-tech users
sed -i 's/A production-grade ML system/A production-grade ML system/' README.md

sed -i 's/## Quick Start (3 Steps)/## 🚀 Quick Start (1 Command)\n\nWe have provided a fully automated setup and execution script:\n\n```bash\n.\/setup_and_run.sh\n```\n\nThis automatically builds containers, starts MLflow, runs the 14-step pipeline, starts the API, and runs tests. For detailed manual steps, see below.\n/' SETUP_GUIDE.md

# Update PRESENTATION_GUIDE.md
sed -i 's/# Presentation Guide/# Presentation Guide\n\n## 📊 System Flow\n\n```mermaid\nsequenceDiagram\n    participant User\n    participant API\n    participant MLflow\n    participant Model\n    User->>API: Send Sensor Data\n    API->>Model: Request Prediction\n    Model-->>API: Stress Level (0=Low, 5=High)\n    API-->>User: JSON Response\n```\n/' PRESENTATION_GUIDE.md

