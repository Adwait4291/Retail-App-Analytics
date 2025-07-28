Here's the updated documentation for `Dockerfile.streamlit`:

```markdown
# Streamlit Dashboard Dockerfile Configuration

This Dockerfile creates the Streamlit web application container for the Retail ML application, providing an interactive dashboard for model monitoring, data visualization, and single predictions.

## Dockerfile.streamlit

```dockerfile
FROM retail-ml-base:latest
```
Inherits from the custom base image `retail-ml-base:latest`, which provides:
- Python 3.12-slim foundation with Streamlit dependencies
- Common system packages and build tools
- Security configurations with non-root user
- Critical environment variables (`DOCKER_CONTAINER=true`, `PYTHONPATH=/app`)

### Temporary Root Access for Setup

```dockerfile
USER root
```
Temporarily switches to root user for directory creation and file copying operations that require elevated permissions.

### Application Code and Configuration

```dockerfile
COPY --chown=app:app app/ /app/app/
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app .env /app/.env
```
Copies essential application files with proper ownership:
- `app/`: Streamlit application code, including dashboard components and single prediction modules
- `src/`: Shared source code for data processing and utilities
- `.env`: Environment configuration file with MongoDB credentials and cluster information
- `--chown=app:app`: Ensures files are owned by the non-root `app` user for security

### Model Storage Setup

```dockerfile
RUN mkdir -p /app/models && \
    chown -R app:app /app/models
```
Creates the model directory structure:
- `/app/models`: Storage for ML model artifacts, scalers, and feature definitions
- **Shared with ML Pipeline**: Models trained by the pipeline are accessible here
- `chown -R app:app`: Ensures proper ownership for the non-root user

### Security Restoration

```dockerfile
USER app
```
Switches back to the non-root `app` user for all subsequent operations, maintaining security best practices.

### Environment Configuration

```dockerfile
ENV MODEL_DIR=/app/models
```
Sets the model directory path explicitly:
- **Critical for Docker environment**: Ensures consistent model path resolution
- **Enables path conversion**: Works with `DOCKER_CONTAINER=true` for automatic path conversion from `../models/` to `/app/models/`
- **Volume mapping compatibility**: Aligns with docker-compose shared volume configuration

### Port Configuration

```dockerfile
EXPOSE 8501
```
Exposes Streamlit's default port 8501:
- **Container port exposure**: Makes the port available for mapping
- **Docker Compose integration**: Maps to dynamic host port via docker-compose
- **Web accessibility**: Enables browser access to the dashboard

### Health Monitoring

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
```
Implements Streamlit-specific health monitoring:
- `--interval=30s`: Checks health every 30 seconds
- `--timeout=10s`: Each health check times out after 10 seconds
- `--start-period=60s`: Allows 60 seconds for Streamlit initialization
- `--retries=3`: Marks container as unhealthy after 3 consecutive failures
- **Streamlit health endpoint**: Uses `/_stcore/health` for accurate application status
- **curl dependency**: Relies on curl installed in the base image

### Application Startup

```dockerfile
CMD ["streamlit", "run", "app/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Defines the Streamlit application startup command:
- `streamlit run`: Launches the Streamlit web server
- `app/streamlit.py`: Main dashboard application file
- `--server.port=8501`: Binds to port 8501 inside the container
- `--server.address=0.0.0.0`: Accepts connections from any IP (required for Docker)

## Dashboard Functionality

### Core Features
- **Model Performance Dashboard**: Real-time display of active model metrics and statistics
- **Data Visualization**: Interactive charts and graphs for data exploration
- **Single Prediction Interface**: User-friendly form for making individual predictions
- **Model Metadata Display**: Information about active models, training dates, and versions
- **Error Handling**: Graceful handling of model loading and prediction errors

### Application Structure
```
/app/app/
├── streamlit.py           # Main dashboard application
├── single_prediction.py   # Single prediction logic with Docker path resolution
├── dashboard/             # Dashboard components
│   ├── overview.py        # Model overview and metrics
│   ├── visualizations.py  # Data visualization components
│   └── predictions.py     # Prediction interface
└── utils/                 # Utility functions
    ├── data_loader.py     # Data loading utilities
    └── formatters.py      # Display formatting functions
```

### Model Integration
- **Shared Volume Access**: Reads models from `/app/models` shared with ML Pipeline
- **Dynamic Model Loading**: Automatically loads the latest active model from MongoDB
- **Path Resolution**: Converts database paths (`../models/`) to Docker paths (`/app/models/`)
- **Real-time Updates**: Reflects new models as they become available

## Usage in Docker Compose

```yaml
streamlit-app:
  build:
    context: .
    dockerfile: Dockerfile.streamlit
  container_name: retail_24-streamlit-app-1
  ports:
    - "8501"                    # Dynamic port mapping
  volumes:
    - models:/app/models        # Shared model access
  environment:
    - DOCKER_CONTAINER=true     # Critical for path resolution
    - MODEL_DIR=/app/models     # Model directory path
    # MongoDB credentials...
  depends_on:
    - mongodb
  restart: unless-stopped
```

### Port Discovery
Since the container uses dynamic port mapping:
```bash
# Find the actual host port
docker-compose port streamlit-app 8501

# Example output: 0.0.0.0:50421
# Access at: http://localhost:50421
```

## Environment Variables

### Inherited from Base Image
- `DOCKER_CONTAINER=true`: **Critical** for model path resolution in `single_prediction.py`
- `PYTHONPATH=/app`: Enables proper module imports
- `USER=app`: Non-root user context

### Container-Specific
- `MODEL_DIR=/app/models`: Explicit model directory path

### Runtime Environment
- MongoDB credentials for database connections
- Streamlit configuration for web server behavior

## Dashboard Components

### Overview Tab
- **Active Model Information**: Version, training date, performance metrics
- **Model Statistics**: F1 Score, accuracy, feature count
- **Data Freshness**: Last update timestamps

### Visualization Tab
- **Feature Importance**: Model feature importance rankings
- **Data Distribution**: Historical data patterns and trends
- **Performance Metrics**: Model performance over time

### Single Prediction Tab
- **User Input Form**: Interactive form for feature input
- **Real-time Predictions**: Instant prediction results with probabilities
- **Feature Influence**: Top features affecting the prediction
- **Result Storage**: Optional saving of predictions to database

## Health Check Details

The Streamlit health check provides:
- **Application Status**: Verifies Streamlit server is running and responsive
- **Web Interface**: Confirms the web interface is accessible
- **Container Health**: Helps orchestration tools manage container lifecycle

Monitor health status:
```bash
# Check health in container logs
docker-compose logs streamlit-app | grep health

# Direct health check
curl http://localhost:$(docker-compose port streamlit-app 8501 | cut -d: -f2)/_stcore/health
```

## Security Features

- **Non-root execution**: All operations run as `app` user
- **File ownership**: Proper ownership of application files
- **Network binding**: Only exposes necessary ports
- **Environment isolation**: Sensitive data passed via environment variables
- **Input validation**: Streamlit provides built-in input sanitization

## Performance Optimization

- **Streamlit Caching**: Uses `@st.cache_data` for expensive operations
- **Model Loading**: Caches loaded models to avoid repeated file I/O
- **Database Connections**: Efficient connection management for MongoDB
- **Resource Management**: Lightweight container with minimal dependencies

## Troubleshooting

### Application Access Issues
```bash
# Find the correct port
docker-compose port streamlit-app 8501

# Check if container is running
docker-compose ps streamlit-app

# View application logs
docker-compose logs -f streamlit-app
```

### Model Loading Problems
```bash
# Verify model files exist
docker exec retail_24-streamlit-app-1 ls -la /app/models

# Test model loading
docker exec retail_24-streamlit-app-1 python -c "
import sys; sys.path.append('/app')
from app.single_prediction import get_active_model_artifacts
model, scaler, features, scaling = get_active_model_artifacts()
print('✅ Success' if model else '❌ Failed')
"
```

### Environment Variable Issues
```bash
# Check critical environment variables
docker exec retail_24-streamlit-app-1 printenv | grep -E "(DOCKER|MODEL|STREAMLIT)"

# Verify Docker environment detection
docker exec retail_24-streamlit-app-1 python -c "
import os
print('Docker detected:', os.environ.get('DOCKER_CONTAINER'))
"
```

### Port and Network Issues
```bash
# Check port mapping
docker port retail_24-streamlit-app-1

# Test network connectivity
docker exec retail_24-streamlit-app-1 curl -f http://localhost:8501/_stcore/health
```

### Health Check Failures
```bash
# Manual health check
curl -f http://localhost:$(docker-compose port streamlit-app 8501 | cut -d: -f2)/_stcore/health

# Check Streamlit server status
docker exec retail_24-streamlit-app-1 ps aux | grep streamlit
```

## Building and Deployment

```bash
# Build the Streamlit image
docker-compose build streamlit-app

# Run Streamlit only
docker-compose up streamlit-app

# Run with logs and dynamic port
docker-compose up streamlit-app --no-deps
docker-compose port streamlit-app 8501
```

## Integration with ML Pipeline

- **Model Synchronization**: Automatically detects new models trained by the pipeline
- **Shared Storage**: Uses Docker volumes for seamless model artifact access
- **Database Coordination**: Reads model metadata from the same MongoDB instance
- **Real-time Updates**: Dashboard reflects pipeline updates without restart

This container provides the user interface for the entire ML system, making trained models accessible through an intuitive web dashboard.
```

## Key Updates Made:

1. **Added Streamlit-specific functionality** - Dashboard components and features
2. **Detailed port configuration** - Dynamic port mapping and discovery
3. **Health check specifics** - Streamlit's `/_stcore/health` endpoint
4. **Application structure** - Complete file organization
5. **Model integration details** - How it works with the ML pipeline
6. **Dashboard component breakdown** - Overview, visualization, and prediction tabs
7. **Performance optimization** - Caching and resource management
8. **Comprehensive troubleshooting** - Streamlit-specific debugging
9. **Security features** - Web application security considerations
10. **Integration documentation** - How it coordinates with other containers

This documentation now fully explains your Streamlit Dockerfile and its role in providing the web interface for your ML system! 🚀