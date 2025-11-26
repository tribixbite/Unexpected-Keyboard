#!/data/data/com.termux/files/usr/bin/bash
# Gradle wrapper that automatically adds custom AAPT2 override for Termux ARM64
# Usage: ./gradle-with-aapt2.sh [gradle args...]

# Detect if we're on Termux ARM64
if [ -f "/data/data/com.termux/files/usr/bin/termux-info" ] && uname -m | grep -q "aarch64\|arm64"; then
    # Termux ARM64 - use custom AAPT2
    AAPT2_PATH="$(pwd)/tools/aapt2-arm64/aapt2"
    if [ -f "$AAPT2_PATH" ]; then
        echo "🔧 Using custom AAPT2 for Termux ARM64: $AAPT2_PATH"
        ./gradlew -Pandroid.aapt2FromMavenOverride="$AAPT2_PATH" "$@"
    else
        echo "⚠️  Custom AAPT2 not found at $AAPT2_PATH, using default"
        ./gradlew "$@"
    fi
else
    # Not Termux ARM64 - use default gradle
    ./gradlew "$@"
fi
