Here's the updated documentation for `Dockerfile.base`:

```markdown
# Base Dockerfile Configuration

This Dockerfile creates a shared base image for all containers in the Retail ML application, providing common dependencies, environment setup, and security configurations.

## Dockerfile.base

```dockerfile
FROM python:3.12-slim
```
Uses Python 3.12 slim image as the base, providing a lightweight foundation with the latest stable Python version for optimal performance and security.

### Working Directory Setup

```dockerfile
WORKDIR /app
```
Sets `/app` as the working directory for all subsequent operations within the container.

### System Dependencies

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
```
Installs essential system packages:
- `build-essential`: Provides compilation tools needed for building Python packages with C extensions
- `curl`: Enables HTTP requests and file downloads within the container
- `--no-install-recommends`: Minimizes image size by avoiding suggested packages
- Cleans up package lists to reduce image size

### Python Dependencies

```dockerfile
COPY requirements.txt .
```
Copies the requirements file from the build context to the container for dependency installation.

```dockerfile
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
```
Python package management:
- Upgrades pip to the latest version for improved dependency resolution
- `--no-cache-dir`: Prevents pip from caching downloaded packages, reducing image size
- Installs all project dependencies from requirements.txt

### Environment Configuration

```dockerfile
ENV PYTHONPATH=/app
ENV DOCKER_CONTAINER=true
```
Sets critical environment variables:
- `PYTHONPATH=/app`: Ensures Python can import modules from the app directory
- `DOCKER_CONTAINER=true`: **Critical for model path resolution** - enables the application to detect Docker environment and convert local paths (`../models/`) to Docker paths (`/app/models/`)

### Security Configuration

```dockerfile
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app
```
Implements security best practices:
- Creates a non-root user named `app` with a home directory and bash shell
- Changes ownership of the `/app` directory to the `app` user
- Switches to the non-root user for all subsequent operations
- Prevents potential security vulnerabilities from running as root

## Usage in Multi-stage Builds

This base image is used by other Dockerfiles in the project:

### Dockerfile.pipeline
```dockerfile
FROM base-image
COPY training/ /app/training/
# Additional pipeline-specific configurations
```

### Dockerfile.streamlit
```dockerfile
FROM base-image
COPY app/ /app/app/
# Additional Streamlit-specific configurations
```

## Key Features

### Environment Detection
The `DOCKER_CONTAINER=true` environment variable is crucial for:
- **Model path resolution** in `single_prediction.py`
- **Automatic path conversion** from local (`../models/`) to Docker (`/app/models/`) paths
- **Seamless operation** across local and containerized environments

### Security
- **Non-root execution**: All processes run as the `app` user
- **Minimal attack surface**: Only essential packages are installed
- **Clean image**: Package caches and unnecessary files are removed

### Performance
- **Slim base image**: Reduces container size and startup time
- **Efficient layering**: Dependencies are installed in separate layers for better caching
- **Optimized pip usage**: No-cache installation reduces final image size

## Build Context Requirements

This Dockerfile expects:
- `requirements.txt` file in the build context root
- Proper Python package specifications in requirements.txt

## Common Environment Variables

When using this base image, applications inherit:
- `PYTHONPATH=/app`: Python module path
- `DOCKER_CONTAINER=true`: Docker environment detection
- `USER=app`: Non-root user context

## Building the Base Image

```bash
# Build the base image
docker build -f Dockerfile.base -t retail-ml-base .

# Use in other Dockerfiles
FROM retail-ml-base
```

## Security Considerations

- **Runs as non-root user**: Enhances container security
- **Minimal package installation**: Reduces potential security vulnerabilities
- **Regular base image updates**: Python 3.12-slim receives security updates
- **Clean package management**: No cached files that could contain vulnerabilities

## Troubleshooting

### Permission Issues
If you encounter permission errors:
```bash
# Check if running as correct user
docker exec container-name whoami
# Should return: app
```

### Path Resolution Issues
Verify the Docker environment detection:
```bash
docker exec container-name printenv DOCKER_CONTAINER
# Should return: true
```

### Import Errors
Check Python path configuration:
```bash
docker exec container-name printenv PYTHONPATH
# Should return: /app
```
```

## Key Updates Made:

1. **Added environment detection explanation** - Detailed explanation of `DOCKER_CONTAINER=true` importance
2. **Enhanced security section** - Comprehensive security features and best practices
3. **Added usage examples** - How this base image is used in multi-stage builds
4. **Added troubleshooting section** - Common issues and solutions
5. **Detailed environment variables** - Explanation of each environment variable's purpose
6. **Performance considerations** - Why specific choices were made
7. **Build context requirements** - What files are needed
8. **Security considerations** - Comprehensive security analysis

This documentation now fully explains your base Dockerfile configuration and its critical role in the Docker environment detection system! 🚀