FROM node:22-bookworm-slim AS chatbot-builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    CHATBOT_PORT=3000 \
    CHATBOT_URL=http://127.0.0.1:3000/api/chat

RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=chatbot-builder /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

COPY src/dashboard/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --from=chatbot-builder /app/package.json ./package.json
COPY --from=chatbot-builder /app/node_modules ./node_modules
COPY src/chatbot ./src/chatbot

COPY .streamlit ./.streamlit
COPY src/dashboard ./src/dashboard
COPY src/helpers ./src/helpers
COPY src/generador_pdf_postmortem.py ./src/generador_pdf_postmortem.py
COPY src/mcp_hana_tools.py ./src/mcp_hana_tools.py
COPY src/workflow.py ./src/workflow.py

RUN chmod +x ./src/dashboard/start.sh

EXPOSE 8501

CMD ["./src/dashboard/start.sh"]
