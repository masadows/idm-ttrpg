@echo off
setlocal enabledelayedexpansion

:: Ustawienie domyślnych wartości
set "LLM_MODEL=qwen3.5:4b"
set "EMBED_MODEL=qwen3-embedding:0.6b"

:: Wczytanie pliku .env, jeśli istnieje
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (.env) do (
        echo %%A | findstr /v /c:"#" >nul
        if not errorlevel 1 (
            if not "%%B"=="" set "%%A=%%B"
        )
    )
)

:: Pobieranie modeli
for %%M in ("%LLM_MODEL%" "%EMBED_MODEL%") do (
    echo ==^> ollama pull %%~M
    ollama pull %%~M
    if errorlevel 1 (
        echo Wystapil blad podczas pobierania modelu %%~M.
        exit /b 1
    )
)

endlocal