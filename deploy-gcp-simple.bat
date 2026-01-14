@echo off
REM Windows용 GCP Cloud Run 간단 배포 스크립트

REM === 여기를 수정하세요 ===
SET PROJECT_ID=YOUR-PROJECT-ID
SET SERVICE_NAME=carbon-ai-backend
SET REGION=asia-northeast3

REM API 키 설정 (Secret Manager 사용 권장)
SET ANTHROPIC_API_KEY=sk-ant-...
SET TAVILY_API_KEY=tvly-...
SET LANGSMITH_API_KEY=lsv2_...
REM ========================

SET IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%:latest

echo 🚀 GCP Cloud Run 배포 시작...

echo 📦 프로젝트 설정...
gcloud config set project %PROJECT_ID%

echo 🔨 Docker 이미지 빌드...
docker build -t %IMAGE_NAME% .

echo 📤 이미지 업로드...
docker push %IMAGE_NAME%

echo 🚢 Cloud Run에 배포...
gcloud run deploy %SERVICE_NAME% ^
  --image %IMAGE_NAME% ^
  --platform managed ^
  --region %REGION% ^
  --allow-unauthenticated ^
  --memory 2Gi ^
  --cpu 1 ^
  --timeout 300 ^
  --port 8080 ^
  --set-env-vars ANTHROPIC_API_KEY=%ANTHROPIC_API_KEY% ^
  --set-env-vars TAVILY_API_KEY=%TAVILY_API_KEY% ^
  --set-env-vars LANGSMITH_API_KEY=%LANGSMITH_API_KEY% ^
  --set-env-vars LANGSMITH_TRACING=true ^
  --set-env-vars LANGSMITH_PROJECT=CarbonAI-Production ^
  --set-env-vars NETZ_MCP_URL=https://hooxi.shinssy.com ^
  --set-env-vars NETZ_MCP_ENABLED=true ^
  --set-env-vars NETZ_ENTERPRISE_ID=1

echo ✅ 배포 완료!
pause
