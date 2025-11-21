#!/data/data/com.termux/files/usr/bin/bash

# This script runs the ONNX hardware acceleration benchmark on a connected Android device.

echo "=== Starting ONNX Benchmark ==="

# Step 1: Clear logcat buffer to ensure we only get logs from this run
 adb logcat -c
echo "Logcat buffer cleared."

# Step 2: Run the specific benchmark test class using the Gradle wrapper.
# The test output will be printed to logcat.
echo "Executing benchmark via Gradle (this may take a few minutes)..."
./gradlew connectedCheck -Pandroid.testInstrumentationRunnerArguments.class=juloo.keyboard2.OnnxBenchmarkTest > /dev/null 2>&1 &

# Display a spinner while Gradle is running
SPINNER='|/-\'
i=0
while ps | grep "gradle" > /dev/null; do
  i=$(( (i+1) %4 ))
  printf "\rRunning... %s" "${SPINNER:$i:1}"
  sleep 0.2
done
printf "\r"
echo "Benchmark execution finished."

# Step 3: Capture and display the benchmark results from logcat.
# We filter by the test's specific tag.
echo
echo "=== Benchmark Results ==="
adb logcat -d -s OnnxBenchmarkTest:I

echo
echo "=== End of Benchmark ==="


