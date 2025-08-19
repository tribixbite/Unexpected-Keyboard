#!/data/data/com.termux/files/usr/bin/bash

# Setup ARM64 Android build tools for Termux
# Downloads and configures ARM64-compatible build tools

echo "=== Setting up ARM64 Build Tools for Termux ==="
echo

ANDROID_HOME="$HOME/android-sdk"
BUILD_TOOLS_VERSION="34.0.0"
ARM64_BUILD_TOOLS="$ANDROID_HOME/build-tools/${BUILD_TOOLS_VERSION}-arm64"

# Check if already exists
if [ -d "$ARM64_BUILD_TOOLS" ]; then
    echo "ARM64 build tools already exist at: $ARM64_BUILD_TOOLS"
    exit 0
fi

echo "Creating ARM64 build tools directory..."
mkdir -p "$ARM64_BUILD_TOOLS"

# Try to get AAPT2 from Termux packages
echo "Looking for Termux-compatible AAPT2..."

# Install aapt and aapt2 if available
pkg install aapt aapt2 -y 2>/dev/null || true

# Check if we have Termux versions
if command -v aapt2 &> /dev/null; then
    echo "Found Termux AAPT2, creating symlinks..."
    ln -sf "$(which aapt2)" "$ARM64_BUILD_TOOLS/aapt2"
    ln -sf "$(which aapt)" "$ARM64_BUILD_TOOLS/aapt" 2>/dev/null || true
fi

# Copy other tools from existing build-tools
EXISTING_BUILD_TOOLS="$ANDROID_HOME/build-tools/35.0.0"
if [ -d "$EXISTING_BUILD_TOOLS" ]; then
    echo "Copying other build tools from $EXISTING_BUILD_TOOLS..."
    
    # Copy tools that might work on ARM64
    for tool in d8 dx zipalign apksigner lib lib64; do
        if [ -e "$EXISTING_BUILD_TOOLS/$tool" ]; then
            cp -r "$EXISTING_BUILD_TOOLS/$tool" "$ARM64_BUILD_TOOLS/" 2>/dev/null || true
        fi
    done
fi

# Create wrapper scripts for problematic x86-64 binaries
echo "Creating wrapper scripts..."

# AAPT2 wrapper that tries multiple approaches
cat > "$ARM64_BUILD_TOOLS/aapt2-wrapper" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# Wrapper for AAPT2 on ARM64

# Try Termux aapt2 first
if command -v aapt2 &> /dev/null; then
    exec aapt2 "$@"
fi

# Fallback: try to use proot if available
if command -v proot &> /dev/null; then
    ORIG_AAPT2="$HOME/android-sdk/build-tools/35.0.0/aapt2"
    if [ -f "$ORIG_AAPT2" ]; then
        exec proot -b /system:/system-x86 "$ORIG_AAPT2" "$@"
    fi
fi

# Last resort: fail with helpful message
echo "Error: No working AAPT2 found for ARM64" >&2
echo "Please install: pkg install aapt2" >&2
exit 1
EOF

chmod +x "$ARM64_BUILD_TOOLS/aapt2-wrapper"

echo
echo "ARM64 build tools setup complete!"
echo "Directory: $ARM64_BUILD_TOOLS"
echo
echo "To use these tools, update your gradle build to use:"
echo "  android.buildToolsVersion = '${BUILD_TOOLS_VERSION}-arm64'"