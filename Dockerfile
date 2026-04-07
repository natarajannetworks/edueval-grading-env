FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH="/app/src"
WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app
EXPOSE 7860
CMD ["uvicorn", "src.envs.grading_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]