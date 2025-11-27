#!/data/data/com.termux/files/usr/bin/bash
#
# Generate code metrics for Unexpected Keyboard project
#

echo "📊 Code Metrics Report - Unexpected Keyboard"
echo "=============================================="
echo ""

echo "📁 Repository Statistics"
echo "------------------------"
echo "Java files: $(find srcs -name "*.java" | wc -l)"
echo "Kotlin files: $(find srcs -name "*.kt" | wc -l)"
echo "Total source files: $(find srcs -name "*.java" -o -name "*.kt" | wc -l)"
echo ""

echo "📏 Lines of Code"
echo "----------------"
java_lines=$(find srcs -name "*.java" -exec cat {} \; | wc -l)
kotlin_lines=$(find srcs -name "*.kt" -exec cat {} \; | wc -l)
total_lines=$((java_lines + kotlin_lines))
echo "Java: $java_lines lines"
echo "Kotlin: $kotlin_lines lines"
echo "Total: $total_lines lines"
echo ""

echo "🏗️ Architecture (Top 10 Largest Files)"
echo "---------------------------------------"
find srcs -name "*.java" -o -name "*.kt" | xargs wc -l | sort -rn | head -11 | tail -10
echo ""

echo "📦 Package Structure"
echo "--------------------"
echo "Main package files: $(ls srcs/juloo.keyboard2/*.java srcs/juloo.keyboard2/*.kt 2>/dev/null | wc -l)"
echo "ML package files: $(find srcs/juloo.keyboard2/ml -name "*.java" -o -name "*.kt" 2>/dev/null | wc -l)"
echo "ONNX package files: $(find srcs/juloo.keyboard2/onnx -name "*.java" -o -name "*.kt" 2>/dev/null | wc -l)"
echo "Prefs package files: $(find srcs/juloo.keyboard2/prefs -name "*.java" -o -name "*.kt" 2>/dev/null | wc -l)"
echo ""

echo "🧪 Test Coverage"
echo "----------------"
test_files=$(find test -name "*.java" 2>/dev/null | wc -l)
test_lines=$(find test -name "*.java" -exec cat {} \; 2>/dev/null | wc -l)
echo "Test files: $test_files"
echo "Test lines: $test_lines"
echo "Test/Source ratio: $(echo "scale=2; $test_lines / $total_lines * 100" | bc)%"
echo ""

echo "📱 APK Information"
echo "------------------"
if [ -f "build/outputs/apk/debug/juloo.keyboard2.debug.apk" ]; then
  apk_size=$(du -h build/outputs/apk/debug/juloo.keyboard2.debug.apk | cut -f1)
  echo "Debug APK size: $apk_size"
fi
echo ""

echo "🔧 Build Configuration"
echo "----------------------"
version_code=$(grep "versionCode" build.gradle | awk '{print $2}')
version_name=$(grep "versionName" build.gradle | awk '{print $2}' | tr -d '"')
echo "Version Code: $version_code"
echo "Version Name: $version_name"
echo ""

echo "📈 Recent Activity"
echo "------------------"
echo "Commits in last 7 days: $(git log --since='7 days ago' --oneline | wc -l)"
echo "Files changed in last commit: $(git diff --name-only HEAD~1 HEAD | wc -l)"
echo ""

echo "✨ Refactoring Progress"
echo "-----------------------"
kb2_lines=$(wc -l srcs/juloo.keyboard2/Keyboard2.java | awk '{print $1}')
echo "Keyboard2.java: $kb2_lines lines (target: <700 lines)"
if [ "$kb2_lines" -lt 700 ]; then
  echo "✅ Refactoring target ACHIEVED!"
else
  remaining=$((kb2_lines - 700))
  echo "⏳ $remaining lines above target"
fi
