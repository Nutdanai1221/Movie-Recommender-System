# Use Python 3.10 as base image
FROM python:3.10

# Set working directory within the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Command to run your application
CMD ["python", "./movie_recommender/app/main.py"]