#!/bin/bash
# Export swipe training data from Android device

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Swipe ML Data Export Tool ===${NC}"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo -e "${RED}Error: No Android device connected${NC}"
    echo "Please connect your device and enable USB debugging"
    exit 1
fi

# Get package name (debug or release)
PACKAGE="juloo.keyboard2.debug"
if ! adb shell pm list packages | grep -q "$PACKAGE"; then
    PACKAGE="juloo.keyboard2"
    if ! adb shell pm list packages | grep -q "$PACKAGE"; then
        echo -e "${RED}Error: Unexpected Keyboard not installed${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}Using package: $PACKAGE${NC}"

# Create output directory
OUTPUT_DIR="data/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Export database
echo "Exporting ML database..."
DB_PATH="/data/data/$PACKAGE/databases/swipe_ml_data.db"
adb shell "run-as $PACKAGE cat $DB_PATH" > "$OUTPUT_DIR/swipe_ml_data.db" 2>/dev/null

if [ ! -s "$OUTPUT_DIR/swipe_ml_data.db" ]; then
    echo -e "${RED}Failed to export database. Trying alternative method...${NC}"
    
    # Try using backup method
    adb backup -f "$OUTPUT_DIR/backup.ab" -noapk $PACKAGE
    
    if [ -f "$OUTPUT_DIR/backup.ab" ]; then
        echo "Backup created. You'll need to extract the database manually."
    else
        echo -e "${RED}Error: Could not export data${NC}"
        echo "Try exporting from the app settings instead"
        exit 1
    fi
else
    echo -e "${GREEN}Database exported successfully${NC}"
    
    # Convert to NDJSON using sqlite3
    if command -v sqlite3 &> /dev/null; then
        echo "Converting to NDJSON format..."
        sqlite3 "$OUTPUT_DIR/swipe_ml_data.db" \
            "SELECT json_data FROM swipe_data ORDER BY timestamp_utc;" | \
            while read -r line; do
                echo "$line" >> "$OUTPUT_DIR/training_data.ndjson"
            done
        
        if [ -f "$OUTPUT_DIR/training_data.ndjson" ]; then
            LINE_COUNT=$(wc -l < "$OUTPUT_DIR/training_data.ndjson")
            echo -e "${GREEN}Exported $LINE_COUNT samples to training_data.ndjson${NC}"
        fi
    else
        echo -e "${YELLOW}sqlite3 not found. Install it to convert to NDJSON${NC}"
    fi
fi

# Try to export from app's external files directory
echo "Checking for exported JSON files..."
EXPORT_PATH="/storage/emulated/0/Android/data/$PACKAGE/files/swipe_ml_export"
adb shell ls "$EXPORT_PATH" 2>/dev/null | while read -r file; do
    if [[ $file == *.json ]] || [[ $file == *.ndjson ]]; then
        echo "Pulling $file..."
        adb pull "$EXPORT_PATH/$file" "$OUTPUT_DIR/" 2>/dev/null
    fi
done

# Get statistics
if [ -f "$OUTPUT_DIR/swipe_ml_data.db" ] && command -v sqlite3 &> /dev/null; then
    echo ""
    echo "Database Statistics:"
    echo "-------------------"
    sqlite3 "$OUTPUT_DIR/swipe_ml_data.db" <<EOF
.mode column
.headers on
SELECT 
    COUNT(*) as total_samples,
    COUNT(DISTINCT target_word) as unique_words,
    SUM(CASE WHEN collection_source = 'calibration' THEN 1 ELSE 0 END) as calibration_samples,
    SUM(CASE WHEN collection_source = 'user_selection' THEN 1 ELSE 0 END) as user_samples
FROM swipe_data;

.headers off
.mode list
SELECT '';
SELECT 'Top 10 words by frequency:';
.mode column
.headers on
SELECT target_word, COUNT(*) as count 
FROM swipe_data 
GROUP BY target_word 
ORDER BY count DESC 
LIMIT 10;
EOF
fi

echo ""
echo -e "${GREEN}Export complete!${NC}"
echo "Data saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review the exported data"
echo "2. Run training: python train_swipe_model.py --data $OUTPUT_DIR/training_data.ndjson"
echo "3. Deploy the trained model back to the app"