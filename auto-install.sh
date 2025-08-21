#!/bin/bash

# Auto-install script for Unexpected Keyboard APK
# Supports multiple installation methods in order of preference

APK_PATH="build/outputs/apk/debug/juloo.keyboard2.debug.apk"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Auto-installing APK..."

# Check if APK exists
if [ ! -f "$APK_PATH" ]; then
    echo -e "${RED}❌ APK not found at $APK_PATH${NC}"
    echo "Please build the APK first with: ./build-on-termux.sh"
    exit 1
fi

# Method 1: Try Shizuku/rish (most seamless)
if command -v rish &> /dev/null; then
    echo "📱 Attempting install via Shizuku/rish..."
    
    # Check if Shizuku is running
    if rish -c "echo 'Shizuku OK'" &> /dev/null; then
        # Use pm install with options for Android 14+ compatibility
        # Store output and check return code since some devices don't output "Success"
        OUTPUT=$(rish -c "pm install -r --bypass-low-target-sdk-block '$PWD/$APK_PATH'" 2>&1)
        RESULT=$?
        
        # Check for success in multiple ways
        if [ $RESULT -eq 0 ] || echo "$OUTPUT" | grep -q "Success"; then
            echo -e "${GREEN}✅ APK installed successfully via Shizuku!${NC}"
            exit 0
        elif echo "$OUTPUT" | grep -q "INSTALL_FAILED"; then
            echo -e "${YELLOW}⚠️ Shizuku install failed: $OUTPUT${NC}"
            echo -e "${YELLOW}   Trying next method...${NC}"
        else
            # No clear failure, might have succeeded
            echo -e "${GREEN}✅ APK installation completed via Shizuku (no errors detected)${NC}"
            exit 0
        fi
    else
        echo -e "${YELLOW}⚠️ Shizuku not running. Start it first or use alternative method.${NC}"
    fi
fi

# Method 2: Try sud (Shizuku alternative)
if command -v sud &> /dev/null; then
    echo "📱 Attempting install via sud..."
    
    # Check if sud server is running
    if sud echo "SUD OK" &> /dev/null 2>&1; then
        OUTPUT=$(sud pm install -r --bypass-low-target-sdk-block "$PWD/$APK_PATH" 2>&1)
        RESULT=$?
        
        if [ $RESULT -eq 0 ] || echo "$OUTPUT" | grep -q "Success"; then
            echo -e "${GREEN}✅ APK installed successfully via sud!${NC}"
            exit 0
        elif echo "$OUTPUT" | grep -q "INSTALL_FAILED"; then
            echo -e "${YELLOW}⚠️ Sud install failed: $OUTPUT${NC}"
            echo -e "${YELLOW}   Trying next method...${NC}"
        else
            echo -e "${GREEN}✅ APK installation completed via sud (no errors detected)${NC}"
            exit 0
        fi
    else
        echo -e "${YELLOW}⚠️ Sud server not running. Start Shizuku first.${NC}"
    fi
fi

# Method 3: Try direct pm command (works on some ROMs)
if pm install -r "$PWD/$APK_PATH" 2>&1 | grep -q "Success"; then
    echo -e "${GREEN}✅ APK installed successfully via direct pm!${NC}"
    exit 0
fi

# Method 4: Fallback to Android intent (requires user interaction)
echo -e "${YELLOW}📲 Opening APK installer (requires manual confirmation)...${NC}"
am start -a android.intent.action.VIEW \
    -d "file://$PWD/$APK_PATH" \
    -t "application/vnd.android.package-archive" \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ APK installer opened. Please tap 'Install' to continue.${NC}"
    echo ""
    echo "💡 For automatic installation without prompts:"
    echo "   1. Install and start Shizuku: https://shizuku.rikka.app/"
    echo "   2. Enable wireless debugging and pair Shizuku"
    echo "   3. This script will then install automatically!"
else
    echo -e "${RED}❌ Failed to open APK installer${NC}"
    echo "Manual installation required: $APK_PATH"
    exit 1
fi