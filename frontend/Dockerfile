# Use the official Streamlit image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY ./src /app/src


# Expose port 8501 to the outside world
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "--server.port", "8501", "src/app.py"]