#!/bin/bash
# Automated Gesture Testing for v1.32.929
# Uses ADB input commands to simulate gestures

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "Automated Gesture Testing v1.32.929"
echo "========================================="
echo ""

# Get screen dimensions
SCREEN_SIZE=$(adb shell wm size | grep "Physical size" | awk '{print $3}')
WIDTH=$(echo $SCREEN_SIZE | cut -d'x' -f1)
HEIGHT=$(echo $SCREEN_SIZE | cut -d'x' -f2)

echo "Screen size: ${WIDTH}x${HEIGHT}"
echo ""

# Keyboard is at bottom of screen
# Approximate keyboard height: 30% of screen
KB_TOP=$((HEIGHT * 70 / 100))
KB_BOTTOM=$HEIGHT
KB_CENTER_Y=$((KB_TOP + (KB_BOTTOM - KB_TOP) / 2))

# Key positions (approximate for QWERTY layout)
# Row 1 (qwerty): y = KB_TOP + 20%
# Row 2 (asdfgh): y = KB_TOP + 40%
# Row 3 (zxcvbn): y = KB_TOP + 60%
# Bottom row (ctrl/fn/backspace): y = KB_TOP + 80%

ROW1_Y=$((KB_TOP + (KB_BOTTOM - KB_TOP) * 20 / 100))
ROW2_Y=$((KB_TOP + (KB_BOTTOM - KB_TOP) * 40 / 100))
ROW3_Y=$((KB_TOP + (KB_BOTTOM - KB_TOP) * 60 / 100))
ROW4_Y=$((KB_TOP + (KB_BOTTOM - KB_TOP) * 80 / 100))

# Key positions on row 2 (asdfgh...) - 'c' is 4th key
# Assuming 10 keys per row, each ~10% wide
KEY_WIDTH=$((WIDTH / 10))

# Positions
# 'c' key (row 3, position 3)
C_KEY_X=$((KEY_WIDTH * 3))
C_KEY_Y=$ROW3_Y

# Backspace (right side, bottom row)
BS_KEY_X=$((WIDTH - KEY_WIDTH))
BS_KEY_Y=$ROW4_Y

# Ctrl (left side, bottom row)
CTRL_KEY_X=$((KEY_WIDTH * 1))
CTRL_KEY_Y=$ROW4_Y

echo "Keyboard layout coordinates:"
echo "  'c' key: ($C_KEY_X, $C_KEY_Y)"
echo "  Backspace: ($BS_KEY_X, $BS_KEY_Y)"
echo "  Ctrl: ($CTRL_KEY_X, $CTRL_KEY_Y)"
echo ""

# Function to perform swipe gesture
do_swipe() {
    local start_x=$1
    local start_y=$2
    local direction=$3
    local distance=${4:-50}  # Default 50px swipe

    local end_x=$start_x
    local end_y=$start_y

    case $direction in
        "nw")
            end_x=$((start_x - distance))
            end_y=$((start_y - distance))
            ;;
        "sw")
            end_x=$((start_x - distance))
            end_y=$((start_y + distance))
            ;;
        "ne")
            end_x=$((start_x + distance))
            end_y=$((start_y - distance))
            ;;
        "se")
            end_x=$((start_x + distance))
            end_y=$((start_y + distance))
            ;;
    esac

    echo -e "${BLUE}Swiping $direction from ($start_x,$start_y) to ($end_x,$end_y)${NC}"
    adb shell input swipe $start_x $start_y $end_x $end_y 100
    sleep 0.5
}

# Setup: Open note app and focus text field
echo "Opening test app..."
adb shell input keyevent KEYCODE_HOME
sleep 1
adb shell am start -n com.google.android.keep/.activities.BrowseActivity > /dev/null 2>&1 || \
adb shell am start -n org.fossify.notes/.activities.MainActivity > /dev/null 2>&1 || \
echo "Using current app (ensure text field is focused)"
sleep 2

# Focus a text input (tap in middle of screen)
echo "Focusing text input..."
adb shell input tap $((WIDTH / 2)) $((HEIGHT / 3))
sleep 1

# Clear logcat
echo "Starting logcat monitoring..."
adb logcat -c
adb logcat -s "Keyboard2:D" "Pointers:D" "InputCoordinator:D" > ~/auto-gesture-test.log 2>&1 &
LOGCAT_PID=$!
sleep 1

echo ""
echo "========================================="
echo "TEST 1: Backspace NW gesture (delete word)"
echo "========================================="
# Type a word first
echo "Typing 'test word'..."
adb shell input text "test\ word"
sleep 1

echo "Performing backspace NW gesture..."
do_swipe $BS_KEY_X $BS_KEY_Y "nw" 80
sleep 1

echo -e "${YELLOW}Expected: 'word' should be deleted${NC}"
echo ""

echo "========================================="
echo "TEST 2: 'c' SW gesture (period) - NO shift"
echo "========================================="
echo "Typing ' hello' then 'c' SW gesture..."
adb shell input text "\ hello"
sleep 0.5
do_swipe $C_KEY_X $C_KEY_Y "sw" 60
sleep 1
echo -e "${YELLOW}Expected: Should type period '.' after hello${NC}"
echo ""

echo "========================================="
echo "TEST 3: Ctrl SW gesture (clipboard)"
echo "========================================="
echo "Performing ctrl SW gesture..."
do_swipe $CTRL_KEY_X $CTRL_KEY_Y "sw" 60
sleep 1
echo -e "${YELLOW}Expected: Should open clipboard switcher${NC}"
sleep 2
# Go back
adb shell input keyevent KEYCODE_BACK
sleep 1
echo ""

# Stop logcat
kill $LOGCAT_PID 2>/dev/null || true

echo "========================================="
echo "AUTO-TEST COMPLETE"
echo "========================================="
echo ""
echo "Logcat saved to: ~/auto-gesture-test.log"
echo ""
echo "Manual verification needed for:"
echo "  1. Backspace NW → deleted 'word'"
echo "  2. 'c' SW → typed period '.'"
echo "  3. Ctrl SW → opened clipboard"
echo ""
echo "Review screen and logcat to verify results"
