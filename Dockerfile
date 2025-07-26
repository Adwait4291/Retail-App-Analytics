# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# **NECESSARY CHANGE**: Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image - Copy only necessary files
FROM python:3.11-slim

WORKDIR /app

# **NECESSARY CHANGE**: Create a non-root user for better security
RUN addgroup --system app && adduser --system --group app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# **NECESSARY CHANGE**: Change ownership of the app directory to the new user
RUN chown -R app:app /app

# Switch to the non-root user
USER app

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "streamlit.py"]