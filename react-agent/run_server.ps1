# LangGraph 서버 실행 스크립트 (PowerShell)

# react-agent 디렉토리로 이동
Set-Location $PSScriptRoot

# langgraph.exe 경로 찾기
$pythonPath = python -c "import sys; print(sys.executable)"
$scriptsPath = $pythonPath -replace "python.exe", "Scripts\langgraph.exe"

# Microsoft Store Python의 경우 다른 경로
if (-not (Test-Path $scriptsPath)) {
    $scriptsPath = python -c "import site; import os; print(os.path.join(site.getusersitepackages().replace('site-packages', 'Scripts'), 'langgraph.exe'))"
}

# langgraph.exe 실행
if (Test-Path $scriptsPath) {
    Write-Host "LangGraph 서버를 시작합니다..." -ForegroundColor Green
    Write-Host "경로: $scriptsPath" -ForegroundColor Yellow
    & $scriptsPath dev
} else {
    Write-Host "langgraph.exe를 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "다음 명령어로 설치하세요: pip install -e `".[dev]`"" -ForegroundColor Yellow
}

