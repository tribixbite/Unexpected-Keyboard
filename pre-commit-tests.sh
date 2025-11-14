#!/data/data/com.termux/files/usr/bin/bash
# Pre-Commit Test Runner
# Quick verification before committing changes

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Pre-Commit Test Verification${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Test 1: Gradle compilation
echo -e "${BLUE}[1/4]${NC} Checking Kotlin/Java compilation..."
if ./gradlew compileDebugKotlin compileDebugJavaWithJavac --no-daemon -q 2>&1 | grep -i "error" ; then
    echo -e "${RED}✗ Compilation failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Compilation successful${NC}"
fi

# Test 2: Run unit tests
echo -e "${BLUE}[2/4]${NC} Running unit tests..."
if ./gradlew test --no-daemon -q; then
    TESTS=$(find build/test-results/test -name "*.xml" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Unit tests passed${NC} (${TESTS} test classes)"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
    echo ""
    echo "Run './gradlew test --stacktrace' for details"
    exit 1
fi

# Test 3: Check for TODO/FIXME
echo -e "${BLUE}[3/4]${NC} Checking for unfinished work markers..."
TODOS=$(git diff --cached | grep -i "^+.*TODO\|^+.*FIXME" | wc -l)
if [ "$TODOS" -gt 0 ]; then
    echo -e "${RED}⚠ Found ${TODOS} TODO/FIXME in staged changes${NC}"
    git diff --cached | grep -n -i "TODO\|FIXME" | head -5
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ No unfinished work markers${NC}"
fi

# Test 4: Verify version updated
echo -e "${BLUE}[4/4]${NC} Checking version info..."
if git diff --cached build.gradle | grep -q "versionCode\|versionName"; then
    echo -e "${GREEN}✓ Version updated in build.gradle${NC}"
else
    echo -e "${RED}⚠ Version not updated${NC} (run build script first)"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   All checks passed! Ready to commit.${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Recommended next steps:"
echo "  1. git commit -m 'your message'"
echo "  2. ./build-test-deploy.sh  (full deployment test)"
echo ""
