# Unexpected Keyboard - Technical Specifications

Complete technical documentation for Unexpected Keyboard features and subsystems.

---

## 📚 Table of Contents

### Core Features

1. **[Dictionary Manager](DICTIONARY_MANAGER.md)** - Dictionary management UI with search, filtering, and word control
   - Multi-source dictionary management (Main 50k, User, Custom)
   - Real-time search with prefix indexing
   - Performance optimizations for large datasets
   - Tab-based interface with result counts

2. **[Swipe Typing](SWIPE_PREDICTION_PIPELINE.md)** - Neural network-based swipe prediction system
   - Complete pipeline: Input → Encoder → Beam Search → Vocabulary
   - ONNX Runtime integration
   - Performance analysis and optimizations
   - Raw/closest predictions display

3. **[Beam Search & Vocabulary](BEAM_SEARCH_VOCABULARY.md)** - Vocabulary filtering and ranking system
   - 50k word vocabulary with frequency-based ranking
   - Hybrid frequency model (main + custom + user dictionaries)
   - Tier system for common word boosting
   - Prefix indexing for fast lookups

### User Input Features

4. **[Swipe Symbols](SWIPE_SYMBOLS.md)** - Short swipe gestures for quick symbol access
   - 8-directional swipe detection
   - 17 two-letter word shortcuts
   - Hit zone configuration
   - Debug logging system

5. **[Auto-Correction](AUTO_CORRECTION.md)** - Fuzzy matching and auto-correction
   - Edit distance algorithms
   - Capitalization preservation
   - Context-aware correction
   - User-controllable weights

---

## 🔧 Quick Links by Topic

### For Developers

**Getting Started:**
- See main [CLAUDE.md](../../CLAUDE.md) for build commands and development workflow
- See [memory/pm.md](../../memory/pm.md) for project management and current status

**Prediction System:**
1. [SWIPE_PREDICTION_PIPELINE.md](SWIPE_PREDICTION_PIPELINE.md) - Overall pipeline architecture
2. [BEAM_SEARCH_VOCABULARY.md](BEAM_SEARCH_VOCABULARY.md) - Vocabulary and ranking details
3. [AUTO_CORRECTION.md](AUTO_CORRECTION.md) - Typing auto-correction

**User Input:**
1. [SWIPE_SYMBOLS.md](SWIPE_SYMBOLS.md) - Swipe gesture shortcuts
2. [DICTIONARY_MANAGER.md](DICTIONARY_MANAGER.md) - Word management UI

### For Users

**Customization:**
- [DICTIONARY_MANAGER.md](DICTIONARY_MANAGER.md#user-workflows) - How to manage words
- [SWIPE_SYMBOLS.md](SWIPE_SYMBOLS.md#complete-symbol-reference) - Available swipe shortcuts

**Understanding Predictions:**
- [SWIPE_PREDICTION_PIPELINE.md](SWIPE_PREDICTION_PIPELINE.md#pipeline-architecture) - How swipe predictions work
- [BEAM_SEARCH_VOCABULARY.md](BEAM_SEARCH_VOCABULARY.md#scoring-algorithm) - How words are ranked

---

## 📊 Current Implementation Status

| Feature | Status | Version | Notes |
|---------|--------|---------|-------|
| Dictionary Manager | ✅ Complete | v1.32.200 | Tab counts, instant search |
| Swipe Prediction | ✅ Complete | v1.32.198 | Raw/closest predictions |
| Beam Search | ✅ Complete | v1.32.183 | 50k vocabulary, prefix indexing |
| Swipe Symbols | ✅ Complete | v1.32.133 | 17 word shortcuts |
| Auto-Correction | ✅ Complete | v1.32.121 | Fuzzy matching |
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

**Last Updated**: 2025-10-22
**Project**: Unexpected Keyboard
**Repository**: https://github.com/Julow/Unexpected-Keyboard
