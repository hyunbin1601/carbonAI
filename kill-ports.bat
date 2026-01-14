@echo off
echo ========================================
echo 포트 사용 프로세스 종료
echo ========================================
echo.

echo 1. 포트 7860 종료...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":7860" ^| find "LISTENING"') do (
    echo PID %%a 종료 중...
    taskkill /F /PID %%a
)

echo 2. 포트 8080 종료...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8080" ^| find "LISTENING"') do (
    echo PID %%a 종료 중...
    taskkill /F /PID %%a
)

echo 3. 포트 10000 종료...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":10000" ^| find "LISTENING"') do (
    echo PID %%a 종료 중...
    taskkill /F /PID %%a
)

echo.
echo ✅ 완료!
pause
