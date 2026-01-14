@echo off
REM Cloud Build를 사용한 GCP Cloud Run 배포 (Docker 불필요)

REM === 프로젝트 설정 ===
SET PROJECT_ID=gen-lang-client-0529157136
SET SERVICE_NAME=carbon-ai-backend
SET REGION=asia-northeast3

REM API 키 (실제 키로 변경하세요)
SET ANTHROPIC_API_KEY=your-anthropic-api-key-here
SET TAVILY_API_KEY=your-tavily-api-key-here
SET LANGSMITH_API_KEY=your-langsmith-api-key-here
REM ========================

echo 🚀 Cloud Build를 사용한 GCP Cloud Run 배포 시작...
echo.

echo 📦 1단계: 프로젝트 설정...
gcloud config set project %PROJECT_ID%
if errorlevel 1 (
    echo ❌ 프로젝트 설정 실패!
    pause
    exit /b 1
)
echo ✅ 프로젝트 설정 완료
echo.

echo 🔧 2단계: 필요한 API 활성화...
echo - Cloud Run API 활성화 중...
gcloud services enable run.googleapis.com --quiet
echo - Cloud Build API 활성화 중...
gcloud services enable cloudbuild.googleapis.com --quiet
echo - Container Registry API 활성화 중...
gcloud services enable containerregistry.googleapis.com --quiet
echo ✅ API 활성화 완료
echo.

echo 🏗️ 3단계: Cloud Build로 이미지 빌드 및 업로드...
echo (이 단계는 5-10분 정도 소요됩니다)
gcloud builds submit --tag gcr.io/%PROJECT_ID%/%SERVICE_NAME%:latest .
if errorlevel 1 (
    echo ❌ 이미지 빌드 실패!
    pause
    exit /b 1
)
echo ✅ 이미지 빌드 완료
echo.

echo 🚢 4단계: Cloud Run에 배포...
gcloud run deploy %SERVICE_NAME% ^
  --image gcr.io/%PROJECT_ID%/%SERVICE_NAME%:latest ^
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

if errorlevel 1 (
    echo ❌ Cloud Run 배포 실패!
    pause
    exit /b 1
)

echo.
echo ✅ 배포 완료!
echo.
echo 🔗 서비스 URL을 확인하세요:
gcloud run services describe %SERVICE_NAME% --region %REGION% --format="value(status.url)"
echo.
pause
