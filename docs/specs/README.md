# Unexpected Keyboard - Technical Specifications

Complete technical documentation for Unexpected Keyboard features and subsystems.

---

## 📚 Table of Contents

### Core Features

1. **[Dictionary Manager](DICTIONARY_MANAGER.md)** - Dictionary management UI with search, filtering, and word control
   - Multi-source dictionary management (Main 50k, User, Custom)
   - Real-time search with prefix indexing
   - Import/export custom words and disabled words (v1.32.306)
   - Performance optimizations for large datasets
   - Tab-based interface with result counts

1a. **[Clipboard Manager](CLIPBOARD_MANAGER.md)** - Clipboard history with pinning and backup
   - Persistent SQLite-based clipboard history
   - Pin important entries (never expire)
   - Expand/collapse multi-line entries (v1.32.308)
   - Import/export for backup and device migration (v1.32.306)
   - Search and filtering (partial implementation)

2. **[Swipe Typing](SWIPE_PREDICTION_PIPELINE.md)** - Neural network-based swipe prediction system
   - Complete pipeline: Input → Encoder → Beam Search → Vocabulary → Autocorrect
   - ONNX Runtime integration
   - Performance analysis and optimizations
   - Enhanced debug logging (3-stage pipeline transparency)

3. **[Beam Search & Vocabulary](BEAM_SEARCH_VOCABULARY.md)** - Vocabulary filtering and ranking system
   - 50k word vocabulary with frequency-based ranking
   - Hybrid frequency model (main + custom + user dictionaries)
   - Tier system for common word boosting
   - Autocorrect for swipe (fuzzy matching custom words)
   - Prefix indexing for fast lookups

### User Input Features

4. **[Typing Predictions](TYPING_PREDICTION.md)** - Prefix-based word prediction for regular typing
   - Prefix matching with O(1) index lookup
   - Context boost with bigram model (⚠️ not yet validated)
   - Logarithmic frequency scaling
   - User-configurable weights (context boost, frequency scale)

5. **[Short Swipe Gestures](SHORT_SWIPE_GESTURES.md)** - Within-key directional swipes for symbols ✨ NEW
   - Complete system documentation with tolerance algorithm
   - Radial tolerance for equal direction support (v1.32.303)
   - Dynamic sizing based on user settings
   - Direction calculation and mapping (16→9 positions)
   - Configuration and debugging guide

6. **[Swipe Symbols](SWIPE_SYMBOLS.md)** - Historical direction mapping analysis
   - Original hit zone issues (NE/SE)
   - Direction-to-index mapping details
   - ⚠️ Note: Tolerance issues fixed in v1.32.303 (see SHORT_SWIPE_GESTURES.md)

7. **[Auto-Correction](AUTO_CORRECTION.md)** - Fuzzy matching and auto-correction (typing + swipe)
   - Typing autocorrect: Edit distance with capitalization preservation
   - Swipe autocorrect: Custom words fuzzy matched against beam candidates
   - Shared configuration (char match threshold)
   - Future: User-configurable fuzzy matching params (v1.33+)

---

## 🔧 Quick Links by Topic

### For Developers

**Getting Started:**
- See main [CLAUDE.md](../../CLAUDE.md) for build commands and development workflow
- See [memory/pm.md](../../memory/pm.md) for project management and current status

**Prediction System:**
1. [SWIPE_PREDICTION_PIPELINE.md](SWIPE_PREDICTION_PIPELINE.md) - Swipe prediction pipeline (neural network)
2. [BEAM_SEARCH_VOCABULARY.md](BEAM_SEARCH_VOCABULARY.md) - Vocabulary filtering and autocorrect for swipe
3. [TYPING_PREDICTION.md](TYPING_PREDICTION.md) - Typing prediction system (prefix matching)
4. [AUTO_CORRECTION.md](AUTO_CORRECTION.md) - Auto-correction (typing + swipe modes)

**User Input:**
1. [SHORT_SWIPE_GESTURES.md](SHORT_SWIPE_GESTURES.md) - Short swipe gesture system (v2.0)
2. [SWIPE_SYMBOLS.md](SWIPE_SYMBOLS.md) - Swipe gesture shortcuts (historical)
3. [DICTIONARY_MANAGER.md](DICTIONARY_MANAGER.md) - Word management UI
4. [CLIPBOARD_MANAGER.md](CLIPBOARD_MANAGER.md) - Clipboard history and backup

### For Users

**Customization:**
- [DICTIONARY_MANAGER.md](DICTIONARY_MANAGER.md#user-workflows) - How to manage words
- [CLIPBOARD_MANAGER.md](CLIPBOARD_MANAGER.md#user-workflows) - How to use clipboard history and backup
- [SHORT_SWIPE_GESTURES.md](SHORT_SWIPE_GESTURES.md#common-layout-patterns) - Available swipe shortcuts and configuration
- **[../NN_SETTINGS_GUIDE.md](../NN_SETTINGS_GUIDE.md)** - ✨ Complete neural network settings guide (v1.32.340+)

**Understanding Predictions:**
- [SWIPE_PREDICTION_PIPELINE.md](SWIPE_PREDICTION_PIPELINE.md#pipeline-architecture) - How swipe predictions work
- [TYPING_PREDICTION.md](TYPING_PREDICTION.md#scoring-algorithm) - How typing predictions work
- [BEAM_SEARCH_VOCABULARY.md](BEAM_SEARCH_VOCABULARY.md#scoring-algorithm) - How words are ranked
- [AUTO_CORRECTION.md](AUTO_CORRECTION.md#swipe-autocorrect-v13207) - How autocorrect works

**Testing & Troubleshooting:**
- **[../TESTING_CHECKLIST.md](../TESTING_CHECKLIST.md)** - ✨ Testing checklist for NN fixes (v1.32.339-340)

---

## 📊 Current Implementation Status

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Dictionary Manager | ✅ Complete | v1.32.306 | Tab counts, instant search, import/export |
| Clipboard Manager | ⚠️ Partial | v1.32.309 | History, pinning, import/export, expand/collapse; search TODO |
| Swipe Prediction | ✅ Complete | v1.32.207 | Autocorrect + debug logging |
| Beam Search | ✅ Complete | v1.32.207 | 50k vocab, autocorrect, prefix indexing |
| Typing Prediction | ⚠️ Partial | v1.0 | Implemented, bigram not validated |
| Short Swipe Gestures | ✅ Complete | v1.32.303 | Radial tolerance, equal direction support |
| Swipe Symbols | 📚 Historical | v1.32.133 | See SHORT_SWIPE_GESTURES.md for current |
| Auto-Correction | ✅ Complete | v1.32.207 | Typing + swipe modes |
| Neural Network | ✅ Complete | v1.20.0 | ONNX Runtime 1.20.0 |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unexpected Keyboard                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input Layer                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Touch Input │  │ Swipe Gesture│  │Short Swipes  │          │
│  │  (Typing)    │  │ (Swipe Type) │  │ (Symbols)    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│  ┌──────▼──────────────────▼──────────────────▼───────┐         │
│  │           Prediction Engine                         │         │
│  │  ┌─────────────────┐  ┌─────────────────┐          │         │
│  │  │  WordPredictor  │  │ OnnxSwipePredict│          │         │
│  │  │  (Typing Pred.) │  │ (Neural Network)│          │         │
│  │  └─────────┬───────┘  └─────────┬───────┘          │         │
│  │            │                      │                  │         │
│  │  ┌─────────▼──────────────────────▼────────┐        │         │
│  │  │    OptimizedVocabulary (50k words)      │        │         │
│  │  │    - Prefix Indexing                    │        │         │
│  │  │    - Frequency Ranking                  │        │         │
│  │  │    - Tier System (Common/Top3k/Rest)   │        │         │
│  │  └─────────────────┬────────────────────────┘        │         │
│  └────────────────────┼────────────────────────────────┘         │
│                       │                                           │
│  ┌────────────────────▼────────────────────────┐                 │
│  │         Dictionary Sources                   │                 │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐      │                 │
│  │  │  Main   │ │  User   │ │  Custom  │      │                 │
│  │  │  (50k)  │ │  Dict   │ │  Words   │      │                 │
│  │  └─────────┘ └─────────┘ └──────────┘      │                 │
│  └──────────────────────────────────────────────┘                 │
│                                                                 │
│  ┌─────────────────────────────────────────────┐                 │
│  │       Dictionary Manager UI                 │                 │
│  │  - Search with prefix indexing              │                 │
│  │  - Filter by source                         │                 │
│  │  - Tab counts (result numbers)              │                 │
│  │  - Add/Edit/Delete custom words             │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Performance Metrics

### Swipe Prediction Pipeline
- **Total Latency**: 30-75ms (target: <100ms) ✅
  - Feature Extraction: 1-5ms
  - Encoder Inference: 20-40ms (NNAPI/QNN) or 50-80ms (CPU)
  - Beam Search Decoding: 10-30ms
  - Vocabulary Filtering: <1ms
- **Memory Usage**: ~15 MB total
  - Encoder Model: ~4 MB
  - Decoder Model: ~3 MB
  - Vocabulary HashMap: ~7 MB
  - Beam Search Buffers: ~1 MB

### Dictionary Manager
- **Search Performance**: <100ms for 50k words ✅
  - Prefix indexing: O(1) lookup for 1-3 char prefixes
  - Reduces iterations from 50k → 100-500 per keystroke
- **Memory**: +2 MB for prefix index (acceptable)
- **UI Updates**: Instant with notifyDataSetChanged()

### Typing Prediction (Regular Keyboard)
- **Prediction Latency**: <100ms (target: <100ms) ✅
  - Prefix index lookup: <1ms
  - Candidate scoring: 1-2ms
  - Total: 2-4ms typical
- **Dictionary Loading**: Async background loading ✅
  - NO UI freeze during language switching
  - Binary format: 5-10x faster than JSON
  - Loading time: ~30-60ms for 50k words
- **User Dictionary Updates**: Instant ✅
  - ContentObserver for system UserDictionary
  - SharedPreferences listener for custom words
  - NO app restart required

---

## ⚡ Performance Optimizations

Complete performance optimization history across perftodos.md, perftodos2.md, and perftodos3.md:

### v1.32.537-539 - Async Loading & Profiling (perftodos3.md)

**Critical Integration (v1.32.539)**:
- ✅ **Asynchronous Dictionary Loading** - DictionaryManager now uses `loadDictionaryAsync()`
  - Previously: Synchronous `loadDictionary()` blocked UI thread during language switching
  - Now: Background thread loading with ExecutorService
  - Impact: NO MORE UI FREEZES during language changes or app startup
  - Implementation: Callback-based async pattern with main thread completion handlers

- ✅ **UserDictionaryObserver Activation** - Auto-updates for user/custom words
  - Previously: Observer built but never started (dead code)
  - Now: `startObservingDictionaryChanges()` called after dictionary loads
  - Impact: User-added words appear INSTANTLY without app restart
  - Monitors: ContentObserver (UserDictionary.Words) + SharedPreferences (custom words)

**System Profiling (v1.32.537)**:
- ✅ **android.os.Trace Integration** - System-level profiling hooks
  - Added to: WordPredictor, AsyncDictionaryLoader, BinaryDictionaryLoader
  - Enables: Perfetto and Android Studio Profiler integration
  - Impact: Deep performance analysis for future optimizations

### Binary Dictionary Format (perftodos2.md Todo 4)

- ✅ **Binary Contractions** - 4x faster contraction loading
  - Format: Pre-built binary with paired/non-paired sections
  - Performance: ~15ms load time (was 60ms JSON parsing)
  - Files: `assets/dictionaries/contractions.bin`

- ✅ **Binary Dictionaries** - 5-10x faster dictionary loading
  - Format: Header + sorted words + frequencies + prefix index
  - Performance: ~30-60ms for 50k words (was 300ms+ with JSON)
  - Files: `assets/dictionaries/*_enhanced.bin`

### Runtime Performance (perftodos2.md Todo 1)

- ✅ **Removed Verbose Logging** - Eliminated 500ms+ prediction latency
  - Root cause: Excessive debug logging on UI-critical path
  - Fix: BuildConfig.ENABLE_VERBOSE_LOGGING guards
  - Impact: Prediction latency reduced from 600ms → <100ms

### All Optimizations Summary

**Loading Performance**:
- Dictionary loading: 5-10x faster (binary format)
- Contraction loading: 4x faster (binary format)
- Language switching: Non-blocking (async loading)
- App startup: Non-blocking (async loading)

**Runtime Performance**:
- Prediction latency: <100ms (was 600ms)
- Dictionary updates: Instant (ContentObserver + SharedPreferences)
- User words: Instant appearance (no restart)

**Profiling**:
- System-level tracing: Perfetto integration
- Performance analysis: Android Studio Profiler support

**Total Completed**: 12/12 tasks across perftodos.md, perftodos2.md, perftodos3.md ✅

---

## 📖 Version History

See individual specification files for detailed changelogs:
- [DICTIONARY_MANAGER.md Changelog](DICTIONARY_MANAGER.md#changelog)
- [BEAM_SEARCH_VOCABULARY.md Changelog](BEAM_SEARCH_VOCABULARY.md#changelog)
- [SWIPE_PREDICTION_PIPELINE.md Changelog](SWIPE_PREDICTION_PIPELINE.md#changelog)

---

## 🤝 Contributing

When adding new features or subsystems:

1. **Create a spec document** in `docs/specs/`
2. **Follow the standard template**:
   - Overview and goals
   - Architecture diagrams
   - Technical implementation
   - Performance requirements
   - Testing strategy
   - Changelog
3. **Update this README** - Add to table of contents
4. **Cross-reference** related specs
5. **Keep specs updated** as implementation evolves

---

## 📝 Document Template

For new specifications, follow this structure:

```markdown
# Feature Name Specification

**Version**: 1.0
**Status**: Planned | In Progress | Implemented
**Last Updated**: YYYY-MM-DD

## Overview
- Goals and non-goals
- User requirements

## Architecture
- Component diagrams
- Data flow

## Technical Implementation
- Data models
- Algorithms
- APIs

## Performance Requirements
- Latency targets
- Memory usage
- Optimization strategies

## Testing
- Test cases
- Known issues

## Changelog
- Version history
```

---

## 📚 External References

- [Android Input Method Framework](https://developer.android.com/guide/topics/text/creating-input-method)
- [ONNX Runtime Android](https://onnxruntime.ai/docs/get-started/with-android.html)
- [RecyclerView Best Practices](https://developer.android.com/guide/topics/ui/layout/recyclerview)
- [Material Design for Android](https://material.io/develop/android)

---

**Last Updated**: 2025-11-11
**Project**: Unexpected Keyboard
**Repository**: https://github.com/Julow/Unexpected-Keyboard
