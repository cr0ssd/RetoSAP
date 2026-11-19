FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY src/dashboard/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY src/dashboard ./src/dashboard
COPY src/helpers ./src/helpers
COPY src/generador_pdf_postmortem.py ./src/generador_pdf_postmortem.py
COPY src/mcp_hana_tools.py ./src/mcp_hana_tools.py

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/dashboard/dashboard.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
