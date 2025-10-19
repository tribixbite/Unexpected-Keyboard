# Project Management - Unexpected Keyboard

**Quick Links**:
- 📋 **[CHANGELOG.md](CHANGELOG.md)** - Complete version history with technical details
- 🧠 **[swipe.md](swipe.md)** - ML/Neural network implementation details
- 📚 **[../CLAUDE.md](../CLAUDE.md)** - Build commands and development workflow

---

## 🔥 Current Status (2025-10-19)

**Latest Version**: v1.32.121 (170)
**Build Status**: ✅ BUILD SUCCESSFUL
**Branch**: feature/swipe-typing

### Recent Work (v1.32.114-121)

**Auto-Correction Feature** - Automatically corrects typos when pressing space
- Fuzzy matching: same length + first 2 letters + 67% positional char match
- Capitalization preservation (teh→the, Teh→The, TEH→THE)
- Smart Termux detection: no space in Termux app, space in other apps
- 4 configurable settings in UI

**WordPredictor Refactor** - Removed legacy swipe fallback system
- Deleted 8 methods (~200 lines) - neural network handles all swipes
- Unified scoring with early fusion (context applied to ALL candidates)
- Removed 8 deprecated settings, added 2 functional weights

**Files**: `WordPredictor.java`, `Config.java`, `Keyboard2.java`, `settings.xml`

See [CHANGELOG.md](CHANGELOG.md) for detailed technical documentation.

---

## 📌 Known Issues

### High Priority
None currently

### Medium Priority
- **Code Organization**: `Keyboard2.java` is 1200+ lines (needs splitting)
- **Documentation**: Some legacy docs need updating

### Low Priority
- Consider adding undo mechanism for auto-correction

---

## 🎯 Next Steps

### Immediate Tasks
1. Test auto-correction in both Termux and normal apps
2. Refactor `Keyboard2.java` into smaller focused files

### Future Enhancements
- Consider ML-based auto-correction (learning from user corrections)
- Improve context model with n-gram support (currently bigram only)
- Add spell-check dictionary for rare/technical words

---

## 🛠️ Quick Reference

### Build Commands
```bash
# Build debug APK
./build-on-termux.sh

# Build release APK
./build-on-termux.sh release

# Install on device
./gradlew installDebug
```

### Git Workflow
```bash
# Status
git status

# Commit
git add -A
git commit -m "type(scope): description"

# View log
git log --oneline -20
```

### Testing
```bash
# Run all tests
./gradlew test

# Check layouts
./gradlew checkKeyboardLayouts
```

---

## 📊 Project Stats

**Lines of Code** (core prediction system):
- `Keyboard2.java`: ~1200 lines (needs refactor)
- `WordPredictor.java`: ~516 lines
- `NeuralSwipeTypingEngine.java`: ~800 lines
- `BigramModel.java`: ~440 lines

**Total**: ~3000 lines of prediction/autocorrect logic

---

## 📝 Development Notes

### Architecture Principles
1. **Neural-first**: ONNX handles all swipe typing, no fallbacks
2. **Early fusion**: Apply context before selecting candidates
3. **App-aware**: Detect Termux app for smart spacing
4. **User control**: All weights configurable via settings

### Code Conventions
- Use conventional commits: `type(scope): description`
- Build and test after every change
- Update CHANGELOG.md for user-facing changes
- Document complex algorithms with inline comments

---

For complete version history and detailed technical documentation, see [CHANGELOG.md](CHANGELOG.md).
