# Use a lightweight Python 3.12 image to match your local environment
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker build caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and models into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit multipage app headlessly
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]