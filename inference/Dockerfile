# Base image with CUDA & PyTorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy everything into container
COPY ./inference/main.py ./main.py
COPY ./fine-tuned-models ./fine-tuned-models

# Install Python packages
RUN pip install --upgrade pip && pip install \
	fastapi \
	transformers \
	peft \
	accelerate \
	torch \
	uvicorn

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


