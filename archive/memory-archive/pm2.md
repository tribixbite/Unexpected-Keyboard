# Project Management: Swipe System Review & Recommendations

This document provides a comprehensive review of the current gesture-to-prediction system and a set of actionable recommendations for improvement.

### System Summary

The project currently contains two parallel swipe-to-word systems:

1.  **Live System: Statistical Classifier**
    *   **Location**: `floris/` directory (`StatisticalGlideTypingClassifier.kt`).
    *   **Method**: A non-ML, geometric approach. It compares the user's swipe path to "ideal" straight-line paths for dictionary words.
    *   **Scoring**: Confidence is calculated based on shape, location, velocity similarity, and word frequency.
    *   **Status**: Currently implemented and active in the app.

2.  **Inactive System: Machine Learning Pipeline**
    *   **Location**: `ml_training/` and `models/` directories.
    *   **Method**: A modern deep learning pipeline using a dual-branch GRU with Attention. It learns from actual swipe data.
    *   **Output**: Produces highly optimized mobile models (`.pte`, `.onnx`).
    *   **Status**: Fully developed but **not integrated** into the Android application.

---

### Errors, Fixes, and Recommendations

Here is a numbered list of findings and recommended actions:

1.  **Primary Issue: The Advanced ML Model is Not Being Used.**
    *   **Error**: The most significant finding is that the sophisticated ML model, which has been trained and prepared for deployment, is completely unused by the application. The app relies solely on the older, less accurate statistical method.
    *   **Recommendation**: Implement a new `MLGlideTypingClassifier.kt`. This class should implement the `GlideTypingClassifier` interface. Instead of geometric calculations, it will load the `.pte` Executorch model (as it's the most recent) and use the PyTorch Android Lite library to perform inference. The `models/mobile_deployment_package_executorch/README_Android_Integration.md` provides a clear guide for this.

2.  **Training Data Underutilization.**
    *   **Observation**: The project contains scripts to generate synthetic data (`generate_sample_data.py`) and has collected real user data (`swipe_data_20250821_235946.json`).
    *   **Recommendation**: Prioritize training on real user data. The training pipeline should be configured to primarily use files like `swipe_data_*.json`. A feedback loop where users can optionally submit their collected data would be invaluable for continuously improving the model.

3.  **Codebase Cleanup and Focus.**
    *   **Observation**: The `ml_training` directory contains multiple training scripts (`train_simple_model.py`, `train_numpy_model.py`, `train_swipe_model.py`, `train_advanced_model.py`). This suggests a history of experimentation.
    *   **Recommendation**: To streamline development, deprecate the older training scripts. The focus should be entirely on `train_advanced_model.py`, as it produces the most sophisticated and accurate model. The older scripts and models can be moved to an `archive/` sub-directory to avoid confusion.

4.  **Lack of Fallback and A/B Testing Mechanism.**
    *   **Issue**: Once the ML model is integrated, if it fails or performs poorly, there is no fallback.
    *   **Recommendation**: Add a developer setting to switch between `StatisticalGlideTypingClassifier` and the new `MLGlideTypingClassifier`. This allows for easy comparison, debugging, and provides a safety net. `GlideTypingManager` can be modified to select the classifier based on this preference.

5.  **Potential UI Thread Blocking.**
    *   **Risk**: ML model inference, even on a mobile-optimized model, can sometimes be slow enough to cause stutter on the UI thread, making the keyboard feel unresponsive.
    *   **Recommendation**: Ensure that the call to the ML model's prediction function happens on a background thread. The `GlideTypingManager` already uses a Coroutine `scope` with `Dispatchers.Default`, which is excellent. This pattern must be strictly maintained for the new ML classifier to prevent UI freezes.

6.  **Refine the Statistical Pruning Algorithm.**
    *   **Observation**: The `Pruner` in the statistical model uses a `findPath` method based on A* search to find likely key sequences. This is computationally intensive.
    *   **Recommendation**: While the primary focus should be on the ML model, the statistical model's `pruneByPath` function could be optimized. Instead of a full A* search across the gesture, a simpler and faster heuristic could be used: find the N-closest keys for the start, middle, and end points of the gesture and prune the dictionary to words that can be formed from those key sets. This would make the fallback model faster.
