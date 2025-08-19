#!/data/data/com.termux/files/usr/bin/bash

# Complete build script for Unexpected Keyboard on Termux ARM64
# This script handles all the compatibility issues

echo "=== Unexpected Keyboard Termux Build Script ==="
echo "This script builds the APK on Termux ARM64 devices"
echo

# 1. Set up environment
export ANDROID_HOME="$HOME/android-sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export JAVA_HOME="/data/data/com.termux/files/usr/lib/jvm/java-17-openjdk"
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/35.0.0:$PATH"

echo "Step 1: Checking prerequisites..."

# Check Java
if ! java -version &>/dev/null; then
    echo "Error: Java not found. Install with: pkg install openjdk-17"
    exit 1
fi

# Check Gradle
if ! gradle -v &>/dev/null; then
    echo "Error: Gradle not found. Install with: pkg install gradle"
    exit 1
fi

# Check Android SDK
if [ ! -d "$ANDROID_HOME" ]; then
    echo "Error: Android SDK not found at $ANDROID_HOME"
    echo "Please install Android SDK first"
    exit 1
fi

echo "Step 2: Installing Termux-compatible build tools..."

# Install AAPT2 if not present
if ! command -v aapt2 &>/dev/null; then
    echo "Installing aapt2..."
    yes | pkg install aapt2 --overwrite '*' 2>/dev/null || {
        echo "Warning: Failed to install aapt2 package"
    }
fi

echo "Step 3: Fixing AAPT2 compatibility..."

# Replace all gradle cached AAPT2 with Termux version
if command -v aapt2 &>/dev/null; then
    TERMUX_AAPT2=$(which aapt2)
    echo "Using Termux AAPT2: $TERMUX_AAPT2"
    
    # Find and replace all AAPT2 binaries in gradle cache
    find ~/.gradle/caches -name "aapt2" -type f 2>/dev/null | while read -r aapt2_file; do
        echo "Replacing: $aapt2_file"
        cp "$TERMUX_AAPT2" "$aapt2_file"
        chmod +x "$aapt2_file"
    done
else
    echo "Warning: Termux AAPT2 not found, build may fail"
fi

echo "Step 4: Creating symbolic link for android.jar..."

# The Termux AAPT2 might have issues with the android.jar path
# Create a workaround by ensuring the path is accessible
ANDROID_JAR="$ANDROID_HOME/platforms/android-35/android.jar"
if [ ! -f "$ANDROID_JAR" ]; then
    echo "Error: android.jar not found at $ANDROID_JAR"
    exit 1
fi

# Ensure the android.jar is readable
chmod +r "$ANDROID_JAR" 2>/dev/null || true

echo "Step 5: Cleaning previous builds..."
./gradlew clean || {
    echo "Warning: Clean failed, continuing anyway..."
}

echo "Step 6: Building Debug APK..."
echo "This may take a few minutes on first run..."

# Build with workarounds for Termux
./gradlew assembleDebug \
    -Dorg.gradle.jvmargs="-Xmx2048m -XX:MaxMetaspaceSize=512m" \
    -Pandroid.aapt2FromMavenOverride="$TERMUX_AAPT2" \
    --no-daemon \
    --warning-mode=all \
    2>&1 | tee build.log

# Check build result
if [ -f "app/build/outputs/apk/debug/app-debug.apk" ]; then
    echo
    echo "=== BUILD SUCCESSFUL! ==="
    echo "APK created at: app/build/outputs/apk/debug/app-debug.apk"
    echo
    ls -lh app/build/outputs/apk/debug/app-debug.apk
    echo
    echo "To install on device:"
    echo "  adb install app/build/outputs/apk/debug/app-debug.apk"
else
    echo
    echo "=== BUILD FAILED ==="
    echo "Check build.log for details"
    echo
    echo "Common issues:"
    echo "1. AAPT2 compatibility - ensure 'pkg install aapt2' succeeded"
    echo "2. Memory issues - try closing other apps"
    echo "3. SDK version mismatch - check Android SDK installation"
    exit 1
fi