# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unexpected Keyboard is a lightweight, privacy-conscious virtual keyboard for Android. The main feature is swipe-to-corner gestures for typing additional characters. Originally designed for programmers using Termux, now suitable for everyday use.

## Build Commands

### Standard Build
```bash
# Build debug APK
./gradlew assembleDebug
# Output: build/outputs/apk/debug/juloo.keyboard2.debug.apk

# Build release APK  
./gradlew assembleRelease
# Output: build/outputs/apk/release/juloo.keyboard2.apk (unsigned)

# Install debug build on connected device
./gradlew installDebug

# Clean build
./gradlew clean
```

### Termux ARM64 Build
For building on Termux (Android ARM64 devices):
```bash
# One-time setup
./setup-arm64-buildtools.sh

# Build using the convenience script
./build-on-termux.sh        # Builds debug APK
./build-on-termux.sh release # Builds release APK

# Or manually with gradle
export ANDROID_HOME="$HOME/android-sdk"
export JAVA_HOME="/data/data/com.termux/files/usr/lib/jvm/java-17-openjdk"
./gradlew assembleDebug
./gradlew assembleRelease
```

**Note**: Termux builds require qemu-x86_64 for AAPT2 emulation. The bundled `tools/aapt2-arm64/` wrapper handles this automatically.

### Test Commands
```bash
# Run all tests
./gradlew test

# Run specific test
./gradlew test --tests "SpecificTestClass.testMethod"

# Check keyboard layouts
./gradlew checkKeyboardLayouts

# Generate layouts list
./gradlew genLayoutsList

# Compile compose sequences (for modifier keys)
./gradlew compileComposeSequences
```

## Development Notes

### Testing Single Test
```bash
./gradlew test --tests "SpecificTestClass.testMethod"
```

### Debug Signing
If you encounter `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, uninstall first:
```bash
adb uninstall juloo.keyboard2.debug
./gradlew installDebug
```

## Architecture

### Core Structure
- **Main Entry**: `Keyboard2.java` - The input method service implementation
- **View Layer**: `Keyboard2View.java` - Custom view handling keyboard rendering and touch events
- **Layout System**: XML-based layouts in `srcs/layouts/` with dynamic loading
- **Configuration**: `Config.java` manages user preferences and keyboard settings
- **Key Processing**: `KeyEventHandler.java` handles key events and modifier states

### Key Components

1. **Layout Management**
   - Layouts defined in XML (`srcs/layouts/*.xml`)
   - `KeyboardData.java` parses and loads layouts
   - `LayoutModifier.java` applies modifiers (Shift, Fn, etc.)
   - Custom layouts supported via `CustomLayoutEditDialog.java`

2. **Gesture System**
   - `Pointers.java` tracks multi-touch and swipe gestures
   - 8-directional swipes (n, ne, e, se, s, sw, w, nw)
   - Anti-clockwise circle gestures

3. **Compose/Modifier System**
   - Compose sequences in `srcs/compose/*.json`
   - `ComposeKey.java` handles compose key combinations
   - Modmap support for custom modifier mappings

4. **Special Features**
   - Clipboard history (`ClipboardHistoryService.java`)
   - Emoji support (`Emoji.java`, `EmojiGridView.java`)
   - Voice input switching (`VoiceImeSwitcher.java`)
   - Auto-capitalization (`Autocapitalisation.java`)

### Resource Generation
Several resources are generated at build time:
- `res/values/layouts.xml` - Generated from layout files via `gen_layouts.py`
- `ComposeKeyData.java` - Generated from compose JSON files
- `assets/special_font.ttf` - Built from SVG files (requires fontforge)
- `res/raw/emojis.txt` - Generated emoji list

### Adding New Features

#### Adding a Layout
1. Create XML file in `srcs/layouts/` following naming: `script_layoutname_locale.xml`
2. Run `./gradlew genLayoutsList` to register
3. Run `./gradlew checkKeyboardLayouts` to validate

#### Adding Key Combinations
1. Edit appropriate JSON in `srcs/compose/`
2. Run `./gradlew compileComposeSequences`

#### Custom Key Values
See `doc/Possible-key-values.md` for built-in key values. Keys can be:
- Characters (outputs verbatim)
- Special functions (shift, fn, ctrl, etc.)
- Locale-specific placeholders (`loc_*`)

## Important Files
- `AndroidManifest.xml` - App configuration and permissions
- `res/xml/method.xml` - Input method configuration and locale support
- `res/values/strings.xml` - UI strings (translatable)
- `build.gradle` - Build configuration and tasks

## Testing
Tests are in `test/juloo.keyboard2/`:
- `KeyValueTest.java` - Key value parsing
- `ComposeKeyTest.java` - Compose sequences
- `ModmapTest.java` - Modifier mappings
- `KeyValueParserTest.java` - Layout parsing

## Python Scripts
Required for development (Python 3):
- `gen_layouts.py` - Generate layout list
- `check_layout.py` - Validate layouts
- `gen_emoji.py` - Generate emoji data
- `srcs/compose/compile.py` - Compile compose sequences

## Contributing Guidelines
- Layouts are CC0 licensed by default
- Use Weblate for translations: https://hosted.weblate.org/engage/unexpected-keyboard/
- Follow existing code patterns for consistency
- Test changes with both debug and release builds
- Ensure compatibility with Android API 21+

## Git Commit Conventions

Use conventional commits format for all commits:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `build`: Build system changes
- `ci`: CI/CD changes
- `perf`: Performance improvements

### Examples
```bash
# Feature
git commit -m "feat(swipe): add swipe typing support"

# Bug fix
git commit -m "fix(layout): correct QWERTY key positioning"

# Documentation
git commit -m "docs: update build instructions for Termux"

# Build changes
git commit -m "build: add ARM64 AAPT2 support for Termux"
```

### Scope Examples
- `layout`: Keyboard layout changes
- `swipe`: Swipe gesture features
- `config`: Configuration/settings
- `ui`: User interface changes
- `build`: Build system
- `test`: Testing infrastructure