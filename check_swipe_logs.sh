#!/bin/bash
echo "=== Checking Key Coordinate Mapping Debug ==="
echo
echo "📍 Key Position Calculations:"
adb logcat -d | grep "📍.*KEY POSITION\|KeyDetection" | tail -20
echo
echo "🔍 Swipe Detection Logs:"
adb logcat -d | grep "🔍\|❌ Too few keys\|✅ SWIPE DETECTED" | tail -10
echo
echo "🎯 Touch Event Logs:"
adb logcat -d | grep "🎯 KEY EVENT\|SwipeDebug" | tail -10
echo
echo "🚨 Main Keyboard Swipe Logs:"  
adb logcat -d | grep "🚨\|🔤 DETECTED KEYS" | tail -10
echo
echo "=== Layout Debug ==="
adb logcat -d | grep "Layout fix\|measured width" | tail -5