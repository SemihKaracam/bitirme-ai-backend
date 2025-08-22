# Base image
FROM python:3.9-slim

# OpenCV için gerekli kütüphaneler
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /yapayzeka

# Önce sadece requirements.txt'i kopyala ve bağımlılıkları kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi tüm proje dosyalarını kopyala (main.py, model, vb.)
COPY . .

# Varsayılan çalıştırma komutu
CMD ["python", "main.py"]




#Eski sürüm

# Base image
# FROM python:3.9-slim

# RUN pip install --upgrade pip

# # Set working directory
# WORKDIR /yapayzeka

# ADD . /yapayzeka
# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Command to run the application
# CMD ["python", "main.py"]
