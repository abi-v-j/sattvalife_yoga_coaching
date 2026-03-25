# 1. Use a slim Python image to keep the build light
FROM python:3.10-slim

# 2. Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    DJANGO_DEBUG=0 \
    DJANGO_ALLOWED_HOSTS=*

# 3. Install System Libraries for OpenCV, Mediapipe, and Audio
# These are essential for your specific requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsndfile1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Install Python dependencies
# We do this before copying the whole project to speed up future builds
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn whitenoise

# 6. Copy the entire project into the container
COPY . .

# 7. Hugging Face Security & Permissions (CRITICAL)
# HF runs containers as a non-root user (UID 1000). 
# We must give this user ownership of the /app folder to run migrations.
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# 8. Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 9. Pre-collect static files so WhiteNoise can serve them immediately
RUN python manage.py collectstatic --noinput

# 10. Expose the port HF expects
EXPOSE 7860

# 11. Run DB migrations, then start the server
# --timeout 120 gives your ML model time to load into RAM without the worker killing itself
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 120 sattvalife_yoga.wsgi:application"]