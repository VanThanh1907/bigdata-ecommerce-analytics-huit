@echo off
REM =============================================================================
REM HUIT Big Data Project - Quick Commands (Windows)
REM Developed by: Lê Văn Thành - HUIT University
REM =============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

echo 🚀 HUIT Big Data Project - Quick Commands
echo =========================================

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="-h" goto help
if "%1"=="--help" goto help

if "%1"=="start" goto start
if "%1"=="pipeline" goto pipeline
if "%1"=="reset" goto reset
if "%1"=="status" goto status

echo ❌ Unknown command: %1
goto help

:start
echo 🌐 Starting HUIT Big Data Web Application...
echo 📊 Dashboard will be available at: http://localhost:5000
echo.
python simple_app.py
goto end

:pipeline
echo ⚙️ Running data pipeline...
echo 📊 This will generate new sample data
echo.
python simple_pipeline.py
echo.
echo ✅ Pipeline completed! Now run: %0 start
goto end

:reset
echo 🔄 Resetting project data...
if exist "data\sample" rmdir /s /q "data\sample" 2>nul
if exist "data\processed" rmdir /s /q "data\processed" 2>nul
echo 📊 Generating fresh data...
python simple_pipeline.py
echo.
echo ✅ Reset completed! Run: %0 start
goto end

:status
echo 📊 Project Status Check:
echo =======================

REM Check if data exists
if exist "data\sample\customers.csv" (
    echo ✅ Sample data exists
    for /f %%i in ('find /c /v "" ^< "data\sample\customers.csv" 2^>nul') do echo    - Customers: %%i records
    for /f %%i in ('find /c /v "" ^< "data\sample\products.csv" 2^>nul') do echo    - Products: %%i records  
    for /f %%i in ('find /c /v "" ^< "data\sample\transactions.csv" 2^>nul') do echo    - Transactions: %%i records
) else (
    echo ❌ No sample data found
    echo    Run: %0 pipeline
)

REM Check if web app is running
curl -s http://localhost:5000 >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Web application is running at http://localhost:5000
) else (
    echo ❌ Web application not running
    echo    Run: %0 start
)

echo.
echo 📁 Project Structure:
if exist "data" (
    dir /b data 2>nul
) else (
    echo    No data directory
)
goto end

:help
echo.
echo 📋 Available Commands:
echo   start     - Khởi động web application
echo   pipeline  - Chạy lại data pipeline (tạo data mới)
echo   reset     - Reset và tạo data mới  
echo   status    - Kiểm tra trạng thái project
echo   help      - Hiện thị hướng dẫn này
echo.
echo 🎯 Typical Workflow:
echo   1. First time: %0 pipeline
echo   2. Then: %0 start
echo   3. Later: just %0 start
echo.
echo 🔗 URLs:
echo   📊 Dashboard: http://localhost:5000
echo   📈 Analytics: http://localhost:5000/api/analytics
echo   📤 Upload: http://localhost:5000/upload
echo.

:end