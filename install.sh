#!/bin/bash

# Quick install script for the last built APK
# Usage: ./install.sh [debug|release]

BUILD_TYPE="${1:-debug}"

if [ "$BUILD_TYPE" = "release" ]; then
    APK_PATH="build/outputs/apk/release/juloo.keyboard2.apk"
else
    APK_PATH="build/outputs/apk/debug/juloo.keyboard2.debug.apk"
fi

# Export for auto-install script
export APK_PATH

# Run auto-installer
./auto-install.sh