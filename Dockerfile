# Use an official Python runtime as a parent image
FROM python:3.9  

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Install any project dependencies (if you have a requirements.txt file)
RUN pip install numpy tensorflow flask

EXPOSE 5000
# Run your Python script
CMD ["python", "main.py"]  
