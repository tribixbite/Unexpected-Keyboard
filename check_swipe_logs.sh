#!/bin/bash
echo "=== Checking Swipe Detection and Prediction Logs ==="
echo
echo "🔍 Swipe Detection Logs:"
adb logcat -d | grep "🔍\|ImprovedSwipeGestureRecognizer\|SWIPE.*DETECTION" | tail -20
echo
echo "🚨 Main Keyboard Swipe Logs:"  
adb logcat -d | grep "🚨\|handleSwipeTyping\|SWIPE PREDICTION" | tail -10
echo
echo "🎯 SwipeTypingEngine Logs:"
adb logcat -d | grep "SwipeTypingEngine\|CGR predictions" | tail -10
echo
echo "📊 KeyboardSwipeRecognizer Logs:"
adb logcat -d | grep "KeyboardSwipeRecognizer" | tail -10
echo
echo "=== Recent All Keyboard Logs ==="
adb logcat -d | grep "Keyboard2\|juloo.keyboard2" | tail -15