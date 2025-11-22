#!/data/data/com.termux/files/usr/bin/bash
#
# Check Termux swipe lag timing
# Monitors timing logs to verify Termux lag fix is working
#

echo "🔍 Termux Swipe Lag Monitor"
echo "================================"
echo ""
echo "Instructions:"
echo "1. Open Termux app"
echo "2. Perform several swipes"
echo "3. This script will show timing breakdown"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

adb logcat -c
adb logcat | grep -E "⏱️|Keyboard2|InputCoordinator" | while read line; do
  if [[ "$line" =~ "⏱️ PREDICTION COMPLETED" ]]; then
    time=$(echo "$line" | grep -oP '\d+ms')
    echo "✅ Prediction: $time"
  elif [[ "$line" =~ "⏱️ UNIFIED DELETE" ]]; then
    time=$(echo "$line" | grep -oP '\d+ms')
    if [[ "$time" =~ ^[0-9]+ms$ ]]; then
      ms=${time%ms}
      if [ "$ms" -lt 50 ]; then
        echo "✅ Deletion: $time (FAST - Fix working!)"
      else
        echo "❌ Deletion: $time (SLOW - Fix may not be working)"
      fi
    fi
  elif [[ "$line" =~ "⏱️ HANDLE_PREDICTIONS COMPLETE" ]]; then
    time=$(echo "$line" | grep -oP '\d+ms')
    echo "✅ Total: $time"
    echo "---"
  fi
done
