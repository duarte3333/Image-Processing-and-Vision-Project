# Makefile for building and running the Docker image

IMAGE_NAME = my-python-app
PYTHON_VERSION = 3.10.12

build:
	@echo "Building Docker image $(IMAGE_NAME):$(PYTHON_VERSION)"
	@sudo docker build -t $(IMAGE_NAME):$(PYTHON_VERSION) .

run:
	@echo "Running Docker container $(IMAGE_NAME):$(PYTHON_VERSION)"
	@sudo docker run -it --rm --name my-running-app $(IMAGE_NAME):$(PYTHON_VERSION)

all: build run

# Add a 'clean' target to remove the Docker image if needed
clean:
	@echo "Removing Docker image $(IMAGE_NAME):$(PYTHON_VERSION)"
	@sudo docker rmi $(IMAGE_NAME):$(PYTHON_VERSION) || true
