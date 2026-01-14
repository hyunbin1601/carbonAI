#!/bin/bash

# GCP Cloud Run 배포 스크립트

# 변수 설정 (YOUR-PROJECT-ID를 실제 프로젝트 ID로 변경)
PROJECT_ID="YOUR-PROJECT-ID"
SERVICE_NAME="carbon-ai-backend"
REGION="asia-northeast3"  # 서울 리전
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "🚀 GCP Cloud Run 배포 시작..."

# 1. GCP 프로젝트 설정
echo "📦 프로젝트 설정: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# 2. Docker 이미지 빌드
echo "🔨 Docker 이미지 빌드..."
docker build -t ${IMAGE_NAME} .

# 3. Container Registry에 푸시
echo "📤 이미지 업로드 중..."
docker push ${IMAGE_NAME}

# 4. Cloud Run에 배포
echo "🚢 Cloud Run에 배포 중..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars "LANGSMITH_TRACING=true" \
  --set-env-vars "LANGSMITH_PROJECT=CarbonAI-Production" \
  --set-env-vars "NETZ_MCP_URL=https://hooxi.shinssy.com" \
  --set-env-vars "NETZ_MCP_ENABLED=true" \
  --set-env-vars "NETZ_ENTERPRISE_ID=1" \
  --set-secrets "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest" \
  --set-secrets "TAVILY_API_KEY=TAVILY_API_KEY:latest" \
  --set-secrets "LANGSMITH_API_KEY=LANGSMITH_API_KEY:latest"

echo "✅ 배포 완료!"
echo "🔗 URL: https://${SERVICE_NAME}-<random-hash>-${REGION}.run.app"
