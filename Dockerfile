# Cloud Run용 Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (빌드 도구 포함)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY react-agent/requirements.txt .
COPY react-agent/setup.py .
COPY react-agent/README.md .

# 의존성 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY react-agent/src ./src
COPY react-agent/knowledge_base ./knowledge_base
COPY react-agent/langgraph.json .

# react-agent 패키지 설치
RUN pip install --no-cache-dir -e .

# 포트 설정 (Cloud Run은 PORT 환경변수 사용)
ENV PORT=8080
EXPOSE 8080

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/ok || exit 1

# 서버 실행
CMD ["python", "-m", "react_agent.server"]
