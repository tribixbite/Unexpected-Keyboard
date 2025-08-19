#!/data/data/com.termux/files/usr/bin/bash

# Fix AAPT2 for Termux builds
# This script replaces the x86-64 AAPT2 binaries with ARM64 compatible ones

echo "Fixing AAPT2 for Termux ARM64..."

# Find all AAPT2 binaries in gradle cache
AAPT2_FILES=$(find ~/.gradle/caches -name "aapt2" -type f 2>/dev/null)

if [ -z "$AAPT2_FILES" ]; then
    echo "No AAPT2 files found in gradle cache"
    exit 1
fi

# Check if we have aapt2 from Termux
TERMUX_AAPT2="/data/data/com.termux/files/usr/bin/aapt2"
if [ ! -f "$TERMUX_AAPT2" ]; then
    echo "Termux AAPT2 not found. Trying to build with bundled tools..."
    
    # Try using Android SDK's aapt2
    SDK_AAPT2="$HOME/android-sdk/build-tools/35.0.0/aapt2"
    if [ -f "$SDK_AAPT2" ]; then
        echo "Found SDK AAPT2 at: $SDK_AAPT2"
        
        # Replace gradle cached AAPT2 with SDK version
        for AAPT2_FILE in $AAPT2_FILES; do
            echo "Replacing: $AAPT2_FILE"
            cp "$SDK_AAPT2" "$AAPT2_FILE"
            chmod +x "$AAPT2_FILE"
        done
    else
        echo "SDK AAPT2 not found either. Please install aapt2 package: pkg install aapt2"
        exit 1
    fi
else
    # Replace with Termux AAPT2
    for AAPT2_FILE in $AAPT2_FILES; do
        echo "Replacing: $AAPT2_FILE"
        cp "$TERMUX_AAPT2" "$AAPT2_FILE"
        chmod +x "$AAPT2_FILE"
    done
fi

echo "AAPT2 fix complete!"