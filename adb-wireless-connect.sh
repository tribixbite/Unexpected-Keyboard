#!/data/data/com.termux/files/usr/bin/bash

# ADB Wireless Connection Script for Termux
# Automatically finds and connects to ADB over WiFi
# Usage: ./adb-wireless-connect.sh [host_ip] [apk_path]

HOST_IP="$1"
APK_PATH="$2"

# Function to find and connect to ADB wireless
connect_adb_wireless() {
    # Save shell's errexit state
    case $- in *e*) was_e=1;; esac
    set +e
    
    # Get host IP from wlan0 or use provided host
    HOST=${1:-$(ifconfig wlan0 2>/dev/null | awk '/inet /{print $2; exit}')}
    
    if [ -z "$HOST" ]; then
        echo "❌ Could not determine wlan0 IP address"
        echo "   Please provide host IP as argument: $0 <host_ip>"
        [ -n "$was_e" ] && set -e
        return 1
    fi
    
    echo "📱 Scanning for ADB on host: $HOST"
    
    # Disconnect any existing connections
    echo "   Disconnecting existing ADB connections..."
    adb disconnect -a >/dev/null 2>&1
    
    # Try standard port first, then scan for open ports
    PORTS="5555"
    
    # Check if nmap is available for port scanning
    if command -v nmap &>/dev/null; then
        echo "   Scanning ports 30000-50000 for ADB..."
        SCANNED_PORTS=$(nmap -p 30000-50000 --open -oG - "$HOST" 2>/dev/null | \
            awk -F"Ports: " '/Ports:/{
                n=split($2,a,/, /); 
                for(i=1;i<=n;i++){ 
                    if (a[i] ~ /open/){ 
                        split(a[i],f,"/"); 
                        print f[1] 
                    } 
                }
            }')
        if [ -n "$SCANNED_PORTS" ]; then
            PORTS="$PORTS $SCANNED_PORTS"
            echo "   Found open ports: $(echo $SCANNED_PORTS | tr '\n' ' ')"
        fi
    else
        echo "   Note: Install nmap for port scanning: pkg install nmap"
    fi
    
    # Try to connect to each port
    for port in $PORTS; do
        echo -n "   Trying $HOST:$port... "
        
        if adb connect "$HOST:$port" >/dev/null 2>&1; then
            # Wait and verify connection
            for i in 1 2 3; do
                sleep 0.5
                if adb devices | grep -q "^$HOST:$port[[:space:]]*device"; then
                    echo "✅ connected!"
                    export ADB_DEVICE="$HOST:$port"
                    [ -n "$was_e" ] && set -e
                    return 0
                fi
            done
            echo "⚠️  failed to verify"
            adb disconnect "$HOST:$port" >/dev/null 2>&1
        else
            echo "❌ no response"
        fi
    done
    
    echo "❌ No working ADB port found on $HOST"
    [ -n "$was_e" ] && set -e
    return 1
}

# Main script
echo "=== ADB Wireless Connection Tool ==="
echo

# Check if adb is installed
if ! command -v adb &>/dev/null; then
    echo "❌ ADB not found. Install with: pkg install android-tools"
    exit 1
fi

# Try to connect
if connect_adb_wireless "$HOST_IP"; then
    echo
    echo "✅ Successfully connected to device at $ADB_DEVICE"
    echo
    
    # Show device info
    echo "📱 Device Information:"
    adb shell getprop ro.product.model 2>/dev/null | sed 's/^/   Model: /'
    adb shell getprop ro.build.version.release 2>/dev/null | sed 's/^/   Android: /'
    adb shell getprop ro.product.manufacturer 2>/dev/null | sed 's/^/   Manufacturer: /'
    echo
    
    # If APK path provided, install it
    if [ -n "$APK_PATH" ] && [ -f "$APK_PATH" ]; then
        echo "📦 Installing APK: $APK_PATH"
        
        # Get package name from APK
        PACKAGE=$(aapt dump badging "$APK_PATH" 2>/dev/null | grep package: | awk '{print $2}' | cut -d"'" -f2)
        
        if [ -n "$PACKAGE" ]; then
            echo "   Package: $PACKAGE"
            echo "   Uninstalling old version..."
            adb uninstall "$PACKAGE" 2>/dev/null || true
        fi
        
        echo "   Installing new version..."
        if adb install -r "$APK_PATH"; then
            echo
            echo "✅ APK installed successfully!"
            
            if [[ "$PACKAGE" == *"keyboard"* ]]; then
                echo
                echo "📝 To enable the keyboard:"
                echo "   1. Go to Settings → System → Languages & input → Virtual keyboard"
                echo "   2. Enable the new keyboard"
                echo "   3. Switch to it using the keyboard selector"
            fi
        else
            echo "❌ Installation failed"
        fi
    else
        echo "💡 Tips:"
        echo "   • To install an APK: $0 $HOST <apk_path>"
        echo "   • To list packages: adb shell pm list packages"
        echo "   • To uninstall: adb uninstall <package_name>"
        echo "   • To disconnect: adb disconnect"
    fi
else
    echo
    echo "❌ Could not establish ADB connection"
    echo
    echo "💡 Troubleshooting:"
    echo "   1. Enable Developer Options on target device"
    echo "   2. Enable 'Wireless debugging' or 'ADB over network'"
    echo "   3. Note the IP address and port shown"
    echo "   4. Make sure both devices are on the same network"
    echo "   5. Try: $0 <device_ip>"
    exit 1
fi