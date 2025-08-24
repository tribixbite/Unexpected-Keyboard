#!/bin/bash
echo "=== Checking SwipeCalibration logs ==="
adb logcat -d | grep "SwipeCalibration" | tail -50
