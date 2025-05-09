FROM python:3.11-slim
WORKDIR /app
ADD . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]