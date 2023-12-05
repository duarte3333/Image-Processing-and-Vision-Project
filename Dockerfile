# Use the official Python image as the base image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# If you have a requirements file, uncomment the next line
RUN apt update; apt install -y libgl1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
CMD ["python3", "main.py", "config/part_1.cfg"]

# sudo docker build -t my-python-app:3.10.12 .
