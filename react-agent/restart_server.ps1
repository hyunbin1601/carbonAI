# LangGraph 서버 재시작 스크립트

Write-Host "기존 LangGraph 프로세스 종료 중..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*langgraph*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process | Where-Object {$_.ProcessName -like "*python*" -and $_.MainWindowTitle -like "*langgraph*"} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

Write-Host "LangGraph 서버 시작 중..." -ForegroundColor Green
cd $PSScriptRoot

# langgraph.exe 직접 실행 시도
$langgraphPath = "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\langgraph.exe"
if (Test-Path $langgraphPath) {
    Write-Host "langgraph.exe를 사용하여 서버 시작" -ForegroundColor Cyan
    Start-Process -FilePath $langgraphPath -ArgumentList "dev" -WorkingDirectory $PSScriptRoot -WindowStyle Normal
} else {
    # Python 모듈로 실행
    Write-Host "python -m langgraph.cli를 사용하여 서버 시작" -ForegroundColor Cyan
    python -m langgraph.cli dev
}

Write-Host "서버가 시작되었습니다. http://localhost:2024 에서 확인하세요." -ForegroundColor Green

