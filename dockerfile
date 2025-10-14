# Usa una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala certificados SSL y herramientas necesarias para compilar dependencias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copia el archivo de dependencias
COPY requirements.txt .

# Instala dependencias de Python, forzando pip a confiar en los hosts de PyPI
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copia el resto del código
COPY . /app

# Expone el puerto de Flask
EXPOSE 5000

# Comando para correr la aplicación Flask en producción
CMD ["python", "flask_app.py"]
