#!/data/data/com.termux/files/usr/bin/bash

# Complete build script for Unexpected Keyboard on Termux ARM64
# This script handles all the compatibility issues
# Usage: ./build-on-termux.sh [debug|release]

BUILD_TYPE="${1:-debug}"
BUILD_TYPE_LOWER=$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')

echo "=== Unexpected Keyboard Termux Build Script ==="
echo "Building $BUILD_TYPE_LOWER APK on Termux ARM64"
echo

# Validate build type
if [[ "$BUILD_TYPE_LOWER" != "debug" && "$BUILD_TYPE_LOWER" != "release" ]]; then
    echo "Error: Invalid build type. Use 'debug' or 'release'"
    echo "Usage: $0 [debug|release]"
    exit 1
fi

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

# Check qemu-x86_64 for AAPT2 wrapper
if ! command -v qemu-x86_64 &>/dev/null; then
    echo "Error: qemu-x86_64 not found. Install with: pkg install qemu-user-x86-64"
    exit 1
fi

echo "Step 2: Preparing layout resources..."

# Ensure layout files are copied (gradle task sometimes doesn't run)
if [ ! -d "build/generated-resources/xml" ] || [ -z "$(ls -A build/generated-resources/xml 2>/dev/null)" ]; then
    echo "Copying layout definitions..."
    mkdir -p build/generated-resources/xml
    cp srcs/layouts/*.xml build/generated-resources/xml/ 2>/dev/null || true
fi

echo "Step 3: Cleaning previous builds..."
./gradlew clean || {
    echo "Warning: Clean failed, continuing anyway..."
}

# Re-copy layouts after clean
mkdir -p build/generated-resources/xml
cp srcs/layouts/*.xml build/generated-resources/xml/ 2>/dev/null || true

# Determine gradle task and output path
if [ "$BUILD_TYPE_LOWER" = "release" ]; then
    echo "Step 4: Building Release APK..."
    echo "Note: Release builds require signing configuration."
    echo "Creating a test signing key for release build..."
    
    # Create a test keystore for release builds if not present
    if [ ! -f "release.keystore" ]; then
        keytool -genkey -v -keystore release.keystore -alias release \
            -keyalg RSA -keysize 2048 -validity 10000 \
            -storepass android -keypass android \
            -dname "CN=Test, OU=Test, O=Test, L=Test, S=Test, C=US" 2>/dev/null || {
            echo "Warning: Could not create release keystore"
        }
    fi
    
    # Set environment variables for release signing
    export RELEASE_KEYSTORE="release.keystore"
    export RELEASE_KEYSTORE_PASSWORD="android"
    export RELEASE_KEY_ALIAS="release"
    export RELEASE_KEY_PASSWORD="android"
    
    GRADLE_TASK="assembleRelease"
    APK_PATH="build/outputs/apk/release/juloo.keyboard2.apk"
else
    GRADLE_TASK="assembleDebug"
    APK_PATH="build/outputs/apk/debug/juloo.keyboard2.debug.apk"
    echo "Step 4: Building Debug APK..."
fi

echo "This may take a few minutes on first run..."

# Build with Termux-specific configuration
./gradlew $GRADLE_TASK \
    -Dorg.gradle.jvmargs="-Xmx2048m -XX:MaxMetaspaceSize=512m" \
    --no-daemon \
    --warning-mode=all \
    2>&1 | tee build-${BUILD_TYPE_LOWER}.log

# Check build result
if [ -f "$APK_PATH" ]; then
    echo
    echo "=== BUILD SUCCESSFUL! ==="
    echo "APK created at: $APK_PATH"
    echo
    ls -lh "$APK_PATH"
    echo
    
    # Auto-install using our new script
    if [ -f "./auto-install.sh" ]; then
        ./auto-install.sh
    elif command -v termux-open &>/dev/null; then
        # Fallback to termux-open if available
        echo "Opening APK for installation..."
        termux-open "$APK_PATH" 2>/dev/null || {
            echo "To install manually, share the APK file to your file manager"
        }
    else
        # Manual instructions as last resort
        echo "To install on device:"
        echo "  1. Share the APK to your file manager"
        echo "  2. Open the APK file to install"
    fi
    
    if [ "$BUILD_TYPE_LOWER" = "release" ]; then
        echo
        echo "Note: Release APK is unsigned. You need to sign it before distribution."
        echo "For testing, you can use debug build instead."
    fi
else
    echo
    echo "=== BUILD FAILED ==="
    echo "Check build-${BUILD_TYPE_LOWER}.log for details"
    echo
    echo "Common issues:"
    echo "1. AAPT2 compatibility - ensure qemu-x86_64 is installed"
    echo "2. Memory issues - try closing other apps"
    echo "3. Missing layouts - check if srcs/layouts/*.xml exist"
    echo "4. SDK version mismatch - check Android SDK installation"
    exit 1
fi