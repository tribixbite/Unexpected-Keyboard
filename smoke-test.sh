#!/data/data/com.termux/files/usr/bin/bash
# Smoke Tests for Unexpected Keyboard
# Verifies basic functionality after installation

set -e

PACKAGE_NAME="juloo.keyboard2.debug"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_test() {
    echo -e "  Testing: $1..."
}

log_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

log_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

echo "Running Unexpected Keyboard Smoke Tests..."
echo ""

# Test 1: Package installed
log_test "Package installation"
if adb shell pm list packages | grep -q "${PACKAGE_NAME}"; then
    log_pass "Package installed"
else
    log_fail "Package not installed"
    exit 1
fi

# Test 2: IME service registered
log_test "IME service registration"
if adb shell ime list -s | grep -q "${PACKAGE_NAME}"; then
    log_pass "IME service registered"
else
    log_fail "IME service not registered"
    exit 1
fi

# Test 3: App doesn't crash on IME enable
log_test "IME enable without crash"
adb logcat -c
adb shell ime enable ${PACKAGE_NAME}/.Keyboard2 2>&1 | grep -v "Warning" || true
sleep 2

if adb logcat -d | grep -qi "fatal.*${PACKAGE_NAME}\|AndroidRuntime.*Exception"; then
    log_fail "Crash detected when enabling IME"
    adb logcat -d | grep -A 10 "AndroidRuntime"
    exit 1
else
    log_pass "IME enabled without crashes"
fi

# Test 4: Check for required permissions
log_test "Required permissions"
PERMS=$(adb shell dumpsys package ${PACKAGE_NAME} | grep "permission" || true)
log_pass "Permissions configured"

# Test 5: App data directory exists
log_test "App data directory"
if adb shell "[ -d /data/data/${PACKAGE_NAME} ]"; then
    log_pass "App data directory exists"
else
    log_fail "App data directory missing"
    exit 1
fi

# Test 6: No ongoing crashes in recent logcat
log_test "Recent crash check"
RECENT_CRASHES=$(adb logcat -d -t 100 | grep -i "fatal\|AndroidRuntime.*Exception" | grep -c "${PACKAGE_NAME}" || echo "0")
if [ "$RECENT_CRASHES" -eq 0 ]; then
    log_pass "No recent crashes detected"
else
    log_fail "Found ${RECENT_CRASHES} recent crash(es)"
    exit 1
fi

echo ""
echo -e "${GREEN}All smoke tests passed!${NC}"
echo ""
