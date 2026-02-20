@echo off
title Automated Experiment Pipeline

echo ============================================================
echo Automated Experiment Pipeline
echo ============================================================
echo.

echo [Step 1/2] Generating test data...
echo ============================================================
python generate_test_data.py
echo.

echo [Step 2/2] Running TotalCaller experiment...
echo ============================================================
python TotalCaller.py
echo.

echo ============================================================
echo Experiment completed!
echo ============================================================
echo.
echo Results saved in: Experiment_Outputs\TotalCaller\
echo.
pause
