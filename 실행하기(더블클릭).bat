@echo off
chcp 65001 >nul
title 럭셔리 상세페이지 생성기 실행기

echo ==============================================
echo 🎨 럭셔리 상세페이지 생성기 실행을 준비합니다...
echo ==============================================
echo.

:: Python 설치 여부 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo 다른 컴퓨터에서 실행하려면 파이썬(Python)이 필요합니다.
    echo https://www.python.org/downloads/ 에 접속하셔서 최신 버전을 설치해주세요.
    echo ※ 설치 시 가장 첫 화면 아래에 있는 [Add python.exe to PATH] 체크박스를 반드시 체크하셔야 합니다!
    echo.
    pause
    exit /b
)

:: 가상 환경(venv)이 없으면 생성
if not exist "venv\" (
    echo [1/3] 전용 독립 환경(venv)을 생성하는 중입니다... (최초 1회만 실행되며 시간이 조금 걸릴 수 있습니다.)
    python -m venv venv
)

:: 가상 환경 실행
call venv\Scripts\activate.bat

:: 필수 라이브러리 설치
echo [2/3] 필요한 프로그램 부품들을 설치 및 점검 중입니다...
pip install -r requirements.txt

:: 프로그램 실행
echo [3/3] 준비가 완료되었습니다! 곧 인터넷 창이 열리며 프로그램이 시작됩니다.
echo.
echo (이 검은색 창은 프로그램을 사용하는 동안 계속 켜두셔야 합니다.)
echo.
python -m streamlit run app.py

pause
