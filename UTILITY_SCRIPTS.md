# Utility Scripts Guide

Collection of helper scripts for Unexpected Keyboard development and testing.

## 🔧 Development Scripts

### Build & Deploy

**`./build-on-termux.sh`**
- Primary build script for Termux ARM64 environment
- Handles custom AAPT2 for ARM64
- Auto-increments version code
- Copies APK to accessible location
- **Usage**: `./build-on-termux.sh` (debug) or `./build-on-termux.sh release`

**`./build-test-deploy.sh`**
- Full pipeline: test → build → deploy → verify
- Runs unit tests before building
- Installs and performs smoke tests
- **Usage**: `./build-test-deploy.sh`

**`./pre-commit-tests.sh`**
- Quick pre-commit verification
- Checks compilation and basic tests
- Faster than full test suite
- **Usage**: `./pre-commit-tests.sh`

### Installation

**`./install-via-adb.sh`**
- Installs APK via ADB wireless
- Handles connection and installation
- **Usage**: `./install-via-adb.sh <path-to-apk>`

**`./adb-wireless-connect.sh`**
- Connects to device wirelessly via ADB
- **Usage**: `./adb-wireless-connect.sh`

## 📊 Monitoring & Analysis

### Performance Monitoring

**`./check_termux_lag.sh`** ⭐ NEW (v1.32.644)
- Real-time Termux swipe lag monitoring
- Shows prediction/deletion/total timing
- Highlights FAST vs SLOW deletion times
- Verifies v1.32.644 lag fix is working
- **Usage**: 
  ```bash
  ./check_termux_lag.sh
  # Then swipe in Termux app
  # Press Ctrl+C to stop
  ```
- **Expected Output**:
  ```
  ✅ Prediction: 45ms
  ✅ Deletion: 8ms (FAST - Fix working!)
  ✅ Total: 53ms
  ---
  ```

**`./run_benchmark.sh`**
- Runs performance benchmarks
- **Usage**: `./run_benchmark.sh`

### Code Analysis

**`./generate_code_metrics.sh`** ⭐ NEW (v1.32.644)
- Comprehensive code statistics
- Java/Kotlin LOC breakdown (46,833 total lines)
- Top 10 largest files
- Package structure analysis
- Test coverage metrics (1% ratio)
- APK size and version info
- Refactoring progress (Keyboard2.java: 692 lines ✅)
- **Usage**: `./generate_code_metrics.sh`

**`./check_app_status.sh`** ⭐ NEW (v1.32.644)
- Checks device connection
- Verifies app installation
- Shows version code/name
- Checks if keyboard enabled/default
- Shows latest APK location
- **Usage**: `./check_app_status.sh`

### Logging

**`./check_swipe_logs.sh`**
- Monitors swipe-related logs
- **Usage**: `./check_swipe_logs.sh`

**`./check_calibration_logs.sh`**
- Monitors calibration logs
- **Usage**: `./check_calibration_logs.sh`

## 🧪 Testing

**`./smoke-test.sh`**
- Post-install verification tests
- Checks app launches without crash
- **Usage**: `./smoke-test.sh`

**`./iterate_neural_test.sh`**
- Neural network iteration testing
- **Usage**: `./iterate_neural_test.sh`

**`./test_neural_iterations.sh`**
- Extended neural testing
- **Usage**: `./test_neural_iterations.sh`

## 🛠️ Maintenance

**`./fix-aapt2.sh`**
- Fixes AAPT2 issues on Termux
- **Usage**: `./fix-aapt2.sh`

**`./install.sh`**
- Simple install wrapper
- **Usage**: `./install.sh`

## 📋 Quick Reference

### Daily Development Workflow

```bash
# 1. Check app status
./check_app_status.sh

# 2. Make code changes
# ... edit files ...

# 3. Build and install
./build-on-termux.sh

# 4. Monitor Termux lag (if testing swipe)
./check_termux_lag.sh

# 5. Generate metrics (optional)
./generate_code_metrics.sh
```

### Troubleshooting

**Device not connected?**
```bash
./adb-wireless-connect.sh
```

**Build failing with AAPT2 error?**
```bash
./fix-aapt2.sh
./build-on-termux.sh
```

**Want to verify lag fix?**
```bash
./check_termux_lag.sh
# Swipe in Termux - look for "FAST" indicator
```

**Check refactoring progress?**
```bash
./generate_code_metrics.sh | grep "Refactoring"
```

## 🎯 New in v1.32.644

Three powerful new utility scripts added:

1. **`check_termux_lag.sh`** - Real-time lag monitoring
   - Verifies 100x speedup from Termux lag fix
   - Shows <10ms deletion times vs previous 900ms

2. **`generate_code_metrics.sh`** - Code statistics
   - Tracks refactoring progress (692 lines, target met!)
   - Shows 46,833 total lines of code
   - 142 source files (104 Java, 38 Kotlin)

3. **`check_app_status.sh`** - Quick status check
   - One command to verify everything
   - Installation, version, keyboard status

## 📚 Documentation

For more details:
- **Build Process**: See `CLAUDE.md` and global `~/.claude/CLAUDE.md`
- **Project Status**: See `memory/pm.md`
- **Performance**: See `memory/perftodos7.md` and `STATE_SUMMARY_v1.32.643.md`
- **Termux Lag Fix**: See `SWIPE_LAG_DEBUG.md`

---

**Tip**: Run `./check_app_status.sh` anytime to see quick actions and current state!
