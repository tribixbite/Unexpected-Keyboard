#!/bin/bash
# Comprehensive Test Suite for v1.32.929
# Tests: Gesture regression fixes + Shift+swipe ALL CAPS feature

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Testing v1.32.929 - Gesture Fixes + ALL CAPS"
echo "========================================="
echo ""

# Function to check test result
check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: $2"
    else
        echo -e "${RED}❌ FAIL${NC}: $2"
    fi
}

# Function to wait for user input
wait_for_test() {
    echo -e "${YELLOW}▶ Test:${NC} $1"
    echo "Press Enter after performing the test..."
    read
}

# 1. Verify installed version
echo "1. Verifying installed version..."
VERSION=$(adb shell dumpsys package juloo.keyboard2.debug | grep "versionName" | head -1 | awk '{print $1}' | cut -d'=' -f2)
if [ "$VERSION" = "1.32.929" ]; then
    check_result 0 "Version $VERSION installed"
else
    check_result 1 "Expected v1.32.929, found $VERSION"
    exit 1
fi
echo ""

# 2. Open test environment
echo "2. Setting up test environment..."
adb shell am start -n com.android.settings/.Settings > /dev/null 2>&1
sleep 1
adb shell input keyevent KEYCODE_HOME
sleep 1

# Open a text editor for testing
echo "Opening test app (Keep Notes)..."
adb shell am start -n com.google.android.keep/.activities.BrowseActivity > /dev/null 2>&1 || \
adb shell am start -n com.android.notes/.NotesListActivity > /dev/null 2>&1 || \
adb shell am start -n com.simplemobiletools.notes.pro/.activities.MainActivity > /dev/null 2>&1 || \
echo "Note: Could not auto-open note app. Please open a text editor manually."
sleep 2
echo ""

# Clear logcat for clean testing
echo "3. Clearing logcat for test monitoring..."
adb logcat -c
adb logcat -s "Keyboard2:D" "Pointers:D" "InputCoordinator:D" > ~/test-v929.log 2>&1 &
LOGCAT_PID=$!
echo "Logcat monitoring started (PID: $LOGCAT_PID)"
echo ""

# Manual tests with prompts
echo "========================================="
echo "REGRESSION TESTS (v1.32.925 fixes)"
echo "========================================="
echo ""

wait_for_test "Shift+c → Should produce 'C' (NOT period '.')"
wait_for_test "Fn+key → Should produce function variant (NOT gesture)"
wait_for_test "Ctrl+key → Should produce control character (NOT gesture)"

echo ""
echo "========================================="
echo "GESTURE TESTS (v1.32.929 fixes)"
echo "========================================="
echo ""

wait_for_test "Backspace NW gesture → Should DELETE LAST WORD"
wait_for_test "Ctrl SW gesture → Should SWITCH CLIPBOARD"
wait_for_test "Fn gesture → Should work correctly"
wait_for_test "'c' key SW gesture (no shift) → Should produce period '.'"

echo ""
echo "========================================="
echo "NEW FEATURE TESTS (v1.32.927)"
echo "========================================="
echo ""

wait_for_test "Normal swipe 'hello' → Should produce 'hello ' (lowercase)"
wait_for_test "Shift+swipe 'hello' → Should produce 'HELLO ' (ALL CAPS)"
wait_for_test "Shift latched + swipe 'test' → Should produce 'TEST ' (ALL CAPS)"
wait_for_test "Shift held + swipe 'world' → Should produce 'WORLD ' (ALL CAPS)"

echo ""
echo "========================================="
echo "TEST COMPLETE"
echo "========================================="
echo ""

# Stop logcat
kill $LOGCAT_PID 2>/dev/null || true

echo "Review logcat at: ~/test-v929.log"
echo ""
echo "Summary:"
echo "- All regression tests should PASS"
echo "- All gesture tests should PASS"
echo "- All shift+swipe tests should produce ALL CAPS"
echo ""
echo "If any test failed, check logcat for details:"
echo "  cat ~/test-v929.log"
