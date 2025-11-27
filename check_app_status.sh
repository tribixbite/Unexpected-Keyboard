#!/data/data/com.termux/files/usr/bin/bash
#
# Check Unexpected Keyboard app status on device
#

echo "📱 Unexpected Keyboard - App Status Check"
echo "=========================================="
echo ""

# Check if ADB is available
if ! command -v adb &> /dev/null; then
    echo "❌ ADB not found. Please ensure Android SDK is installed."
    exit 1
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "❌ No device connected via ADB"
    echo ""
    echo "💡 To connect wirelessly:"
    echo "   ./adb-wireless-connect.sh"
    exit 1
fi

echo "✅ Device connected"
echo ""

# Check if app is installed
if adb shell pm list packages | grep -q "juloo.keyboard2"; then
    echo "✅ App installed: juloo.keyboard2"
    
    # Get version info
    version_code=$(adb shell dumpsys package juloo.keyboard2 | grep versionCode | head -1 | awk '{print $1}' | cut -d'=' -f2)
    version_name=$(adb shell dumpsys package juloo.keyboard2 | grep versionName | head -1 | awk '{print $1}' | cut -d'=' -f2)
    
    echo "   Version Code: $version_code"
    echo "   Version Name: $version_name"
    echo ""
    
    # Check if keyboard is enabled
    if adb shell ime list -s | grep -q "juloo.keyboard2"; then
        echo "✅ Keyboard enabled in system"
        
        # Check if it's the default
        current_ime=$(adb shell settings get secure default_input_method)
        if echo "$current_ime" | grep -q "juloo.keyboard2"; then
            echo "✅ Set as default keyboard"
        else
            echo "⚠️  Enabled but not default. Current: $current_ime"
        fi
    else
        echo "❌ Keyboard not enabled in system"
        echo "   Please enable in Settings → Language & Input"
    fi
    
    echo ""
    
    # Check APK file
    local_apk="/storage/emulated/0/unexpected/debug-kb.apk"
    if adb shell "[ -f $local_apk ] && echo exists" | grep -q exists; then
        apk_size=$(adb shell ls -lh $local_apk | awk '{print $5}')
        echo "📦 Latest APK: $local_apk ($apk_size)"
    fi
    
else
    echo "❌ App not installed"
    echo ""
    echo "💡 To install:"
    echo "   ./build-on-termux.sh"
    echo "   adb install -r /storage/emulated/0/unexpected/debug-kb.apk"
fi

echo ""
echo "🔍 Quick Actions:"
echo "  ./build-on-termux.sh          - Build & install"
echo "  ./check_termux_lag.sh         - Monitor Termux lag"
echo "  ./generate_code_metrics.sh    - Show code metrics"
