# ============================================================================
# Silent Dropout Detection System - optional Docker deployment
# The Docker image is a convenience wrapper; native installation (without
# Docker) is the primary supported path and does not require Docker.
# ============================================================================

# Build stage: install Python deps into a scratch-free wheel house, then
# assemble a slim runtime image.  CPU-only to keep startup fast on low-end
# hardware and to avoid unnecessary GPU/CUDA initialisation.
FROM python:3.10-slim AS build

WORKDIR /app

# Install only the system packages that pip needs at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy just the installed packages (no build tools needed at runtime).
COPY --from=build /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code (excluding data, which is mounted/configured at runtime
# via a named volume or a bind mount).
COPY backend ./backend
COPY ml ./ml
COPY data ./data
COPY .env.example ./.env

# Optional: bind-mount a downloaded dataset rather than baking it into the
# image (keeps the image small and respects licensed download restrictions).
VOLUME ["/data"]

# Environment variables (see .env.example)
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV CORS_ORIGINS=*
ENV API_ADMIN_KEY=change-me

EXPOSE 8000

# Run the API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
