@echo off
echo ===============================================
echo    Heston Stochastic Volatility Model - C++
echo ===============================================
echo.

echo Compiling with optimizations...
g++ -std=c++17 -O3 -Wall -Wextra -march=native -o heston_model.exe main.cpp

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Compilation successful!
    echo.
    echo Starting Heston Model Application...
    echo.
    heston_model.exe
) else (
    echo.
    echo Compilation failed! Please check for errors.
    echo Make sure you have a C++17 compatible compiler installed.
    echo.
    pause
)

echo.
echo Program finished.
pause