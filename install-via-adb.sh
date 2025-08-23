#!/data/data/com.termux/files/usr/bin/bash

# Quick APK installer with ADB wireless auto-connection
# Usage: ./install-via-adb.sh [apk_path] [host_ip]

APK_PATH="${1:-/sdcard/unexpected/debug-kb.apk}"
HOST_IP="$2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== APK Installer with ADB Auto-Connect ===${NC}"
echo

# Check if APK exists
if [ ! -f "$APK_PATH" ]; then
    echo -e "${RED}❌ APK not found: $APK_PATH${NC}"
    echo "Please provide a valid APK path"
    exit 1
fi

echo -e "${GREEN}📦 APK: $APK_PATH${NC}"
ls -lh "$APK_PATH"
echo

# Check if adb is installed
if ! command -v adb &>/dev/null; then
    echo -e "${YELLOW}Installing android-tools...${NC}"
    pkg install -y android-tools
fi

# Function to connect to ADB wireless
adb_wireless_connect() {
    # Save shell options
    case $- in *e*) was_e=1;; esac
    set +e
    
    # Get host IP
    if [ -n "$1" ]; then
        HOST="$1"
    else
        # Try to get wlan0 IP
        HOST=$(ifconfig 2>/dev/null | awk '/wlan0/{getline; if(/inet /) print $2}')
        
        # Fallback to any non-loopback interface
        if [ -z "$HOST" ]; then
            HOST=$(ifconfig 2>/dev/null | awk '/inet / && !/127.0.0.1/{print $2; exit}')
        fi
    fi
    
    if [ -z "$HOST" ]; then
        echo -e "${RED}Could not determine network IP${NC}"
        echo "Please provide host IP: $0 <apk> <host_ip>"
        echo "Example: $0 $APK_PATH 192.168.1.100"
        [ -n "$was_e" ] && set -e
        return 1
    fi
    
    echo -e "${YELLOW}Scanning for ADB on: $HOST${NC}"
    
    # Disconnect existing
    adb disconnect -a >/dev/null 2>&1
    
    # Build port list
    PORTS="5555"
    
    # Quick port scan if nmap available
    if command -v nmap &>/dev/null; then
        echo "Scanning common ADB ports..."
        # Faster scan of common ports
        SCANNED=$(nmap -p 5555,30000-32000,37000-44000 --open -oG - "$HOST" 2>/dev/null | \
            awk -F"Ports: " '/Ports:/{
                n=split($2,a,/, /); 
                for(i=1;i<=n;i++){ 
                    if (a[i] ~ /open/){ 
                        split(a[i],f,"/"); 
                        print f[1] 
                    } 
                }
            }')
        [ -n "$SCANNED" ] && PORTS="$PORTS $SCANNED"
    fi
    
    # Try each port
    for port in $PORTS; do
        echo -ne "  Trying $HOST:$port... "
        
        if timeout 3 adb connect "$HOST:$port" >/dev/null 2>&1; then
            # Verify connection
            sleep 0.5
            if adb devices | grep -q "$HOST:$port.*device"; then
                echo -e "${GREEN}✅ Connected!${NC}"
                export CONNECTED_DEVICE="$HOST:$port"
                [ -n "$was_e" ] && set -e
                return 0
            fi
            adb disconnect "$HOST:$port" >/dev/null 2>&1
        fi
        echo -e "${RED}✗${NC}"
    done
    
    echo -e "${RED}No ADB device found${NC}"
    [ -n "$was_e" ] && set -e
    return 1
}

# Try to connect
if adb_wireless_connect "$HOST_IP"; then
    echo
    echo -e "${GREEN}✅ Connected to: $CONNECTED_DEVICE${NC}"
    
    # Get device info
    MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r\n')
    ANDROID=$(adb shell getprop ro.build.version.release 2>/dev/null | tr -d '\r\n')
    echo -e "📱 Device: ${MODEL} (Android ${ANDROID})"
    echo
    
    # Extract package name
    if command -v aapt &>/dev/null; then
        PACKAGE=$(aapt dump badging "$APK_PATH" 2>/dev/null | grep "package:" | sed "s/.*name='\([^']*\)'.*/\1/")
    else
        # Fallback for debug builds
        if [[ "$APK_PATH" == *"debug"* ]]; then
            PACKAGE="juloo.keyboard2.debug"
        else
            PACKAGE="juloo.keyboard2"
        fi
    fi
    
    # Uninstall old version
    if [ -n "$PACKAGE" ]; then
        echo -e "${YELLOW}Uninstalling old version: $PACKAGE${NC}"
        adb uninstall "$PACKAGE" 2>/dev/null || true
    fi
    
    # Install new APK
    echo -e "${YELLOW}Installing APK...${NC}"
    if adb install -r "$APK_PATH"; then
        echo
        echo -e "${GREEN}✅ Installation successful!${NC}"
        echo
        echo "To enable Unexpected Keyboard:"
        echo "  1. Settings → System → Languages & input"
        echo "  2. Virtual keyboard → Manage keyboards"
        echo "  3. Enable 'Unexpected Keyboard'"
        echo "  4. Use keyboard selector to switch"
    else
        echo -e "${RED}❌ Installation failed${NC}"
        exit 1
    fi
else
    echo
    echo -e "${RED}Failed to connect via ADB${NC}"
    echo
    echo "To enable wireless ADB on target device:"
    echo "  1. Enable Developer Options"
    echo "  2. Enable 'Wireless debugging' or 'ADB over network'"
    echo "  3. Note the IP and port"
    echo "  4. Run: $0 $APK_PATH <device_ip>"
    exit 1
fi