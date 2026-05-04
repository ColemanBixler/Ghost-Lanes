# 1. Explicitly use amd64
FROM --platform=linux/amd64 python:3.7-slim

# 2. Install modern dependencies
RUN apt-get update && apt-get install -y \
    libpng16-16 \
    libtiff6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 3. OVERWRITE and FORCE the symlink
# We use -sf to overwrite 'File exists' errors
RUN ln -sf /usr/lib/x86_64-linux-gnu/libtiff.so.6 /usr/lib/x86_64-linux-gnu/libtiff.so.5 && \
    ldconfig

# 4. Sideload libjpeg8 (Verified stable mirror)
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/libj/libjpeg-turbo/libjpeg-turbo8_2.1.2-0ubuntu1_amd64.deb \
    && dpkg -i libjpeg-turbo8_2.1.2-0ubuntu1_amd64.deb \
    && rm libjpeg-turbo8_2.1.2-0ubuntu1_amd64.deb

WORKDIR /app
COPY . .

# 5. Environment & Python Setup
ENV PYTHONPATH=/app/carla-0.9.13-py3.7-linux-x86_64.egg
ENV CARLA_HOST=host.docker.internal

RUN pip install --no-cache-dir numpy opencv-python torch matplotlib

# 6. Verify the link is valid before finishing build
RUN ls -la /usr/lib/x86_64-linux-gnu/libtiff.so.5
