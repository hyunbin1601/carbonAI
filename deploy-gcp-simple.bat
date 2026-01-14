@echo off
REM Windows용 GCP Cloud Run 간단 배포 스크립트

REM === 여기를 수정하세요 ===
SET PROJECT_ID=gen-lang-client-0529157136
SET SERVICE_NAME=carbon-ai-backend
SET REGION=asia-northeast3

REM API 키 설정 (실제 키로 변경하세요)
SET ANTHROPIC_API_KEY=your-anthropic-api-key-here
SET TAVILY_API_KEY=your-tavily-api-key-here
SET LANGSMITH_API_KEY=your-langsmith-api-key-here
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
  --port 7860 ^
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
