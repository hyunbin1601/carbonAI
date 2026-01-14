@echo off
echo ========================================
echo Hugging Face Spaces 간단 배포
echo ========================================
echo.

SET HF_TOKEN=your-huggingface-token-here


cd C:\ruffy

echo 1. 기존 폴더 삭제...
if exist hf-space rmdir /s /q hf-space

echo 2. Hugging Face Space clone...
git clone https://%HF_TOKEN%@huggingface.co/spaces/ruffy1601/carbon-ai-chatbot hf-space
cd hf-space

echo 3. 필요한 파일 복사...
xcopy /E /I /Y ..\carbon_ai_ver3\react-agent react-agent
copy /Y ..\carbon_ai_ver3\Dockerfile .
copy /Y ..\carbon_ai_ver3\.spacesconfig.yaml .

echo 4. README.md 생성...
(
echo # CarbonAI Chatbot
echo.
echo 탄소 배출권 전문 AI 챗봇
) > README.md

echo 5. .env 파일 생성...
(
echo ANTHROPIC_API_KEY=your-anthropic-api-key-here
echo TAVILY_API_KEY=your-tavily-api-key-here
echo LANGSMITH_API_KEY=your-langsmith-api-key-here
echo LANGSMITH_TRACING=true
echo LANGSMITH_PROJECT=CarbonAI-Production
echo NETZ_MCP_URL=https://hooxi.shinssy.com
echo NETZ_MCP_ENABLED=true
echo NETZ_ENTERPRISE_ID=1
echo PORT=7860
) > react-agent/.env

echo 6. Git add, commit, push...
git add .
git commit -m "Deploy CarbonAI chatbot"
git remote set-url origin https://%HF_TOKEN%@huggingface.co/spaces/ruffy1601/carbon-ai-chatbot
git push origin main

echo.
echo ========================================
echo ✅ 배포 완료!
echo ========================================
echo.
echo URL: https://huggingface.co/spaces/ruffy1601/carbon-ai-chatbot
echo Logs 탭에서 빌드 확인하세요!
echo.

cd ..\carbon_ai_ver3
pause
