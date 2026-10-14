FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema operativo (Compiladores C++, OpenJDK 17 para PaDEL y librerías C++ de RDKit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala el paquete (pyproject.toml) con sus dependencias para aprovechar la
# caché de capas: cambios en el resto del repo no re-descargan las dependencias.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY . .

# El entrenamiento necesita la matriz de datos; data/ y models/ se montan en
# tiempo de ejecucion (volumen) porque .dockerignore los excluye de la imagen.
CMD ["mldd", "train", "target=maob", "task=regression"]