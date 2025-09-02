# Project Memory: Swipe Prediction System Analysis & TODO

This document contains a detailed analysis of the production swipe-to-word prediction system and a list of actionable items for improvement. This is a revised analysis based on the correct source files in `srcs/juloo.keyboard2/`.

---

## High-Level Workflow & Key Files

The production swipe system is a complex, multi-stage pipeline that uses several different strategies to get the best prediction.

1.  **Gesture Recognition (`EnhancedSwipeGestureRecognizer.java`, `Keyboard2View.java`)**: The view captures `MotionEvent`s. The recognizer determines if it's a valid swipe, filters out noise, and tracks the path and keys touched.

2.  **Orchestration (`Keyboard2.java`, `SwipeTypingEngine.java`)**: `Keyboard2` receives the completed swipe data. It passes this to the `SwipeTypingEngine`, which acts as a high-level manager for the prediction process.

3.  **Swipe Classification (`SwipeDetector.java`)**: The engine first uses the `SwipeDetector` to classify the gesture's quality (e.g., `HIGH`, `MEDIUM`, `LOW`). This classification determines which prediction strategy to use.

4.  **Prediction Strategies (`SwipeTypingEngine.java`)**:
    *   **High/Medium Quality Swipes**: A hybrid approach is used, combining results from `DTWPredictor.java` (Dynamic Time Warping) and `WordPredictor.java` (a key sequence-based matcher).
    *   **Low Quality Swipes**: An "enhanced sequence prediction" is used, relying more heavily on the sequence of keys near the path.

5.  **Scoring (`SwipeScorer.java`)**: Candidates from all predictors are fed into the `SwipeScorer`, which calculates a final confidence score. This score is a weighted product of **shape**, **location**, **velocity**, and **word frequency**.

6.  **Pruning (`SwipePruner.java`)**: Before running expensive algorithms like DTW, the `SwipePruner` narrows down the dictionary by looking at the start/end keys of the swipe and the overall path length.

---

## 1. Discrepancies & Architectural Concerns

-   **Multiple Overlapping Systems**: The codebase contains several distinct prediction methodologies: a DTW-based system, a simple key-sequence matcher, and the more advanced `KeyboardSwipeRecognizer` which uses a Bayesian approach. The `SwipeTypingEngine` attempts to blend these, but it creates significant complexity and potential for inconsistent results.
-   **Unused ML Model**: The most significant discrepancy is the presence of a pre-trained ML model (`models/.../swipe_model_android.pte`) and an entire ML data collection/training pipeline (`srcs/juloo/keyboard2/ml/`). However, the production `SwipeTypingEngine` **does not appear to use this ML model for prediction**. It relies entirely on the classic statistical/algorithmic methods (DTW, sequence matching). The ML system is only used for data collection and offline analysis in the `SwipeCalibrationActivity`.
-   **Calibration Screen is a Dev Tool**: The "calibration" screen is not for users. It's a powerful developer tool for collecting training data and debugging the various prediction algorithms. This is a major departure from what its name implies.

---

## 2. Areas for Improvement & Edge Cases

-   **Unify the Prediction Model**: The current hybrid approach is complex and fragmented. The project should commit to a single, more robust prediction strategy. Given the existing investment in an ML pipeline, migrating fully to the neural network model is the clear path forward.
-   **DTW Performance**: Dynamic Time Warping is computationally expensive (O(N*M)) and is a likely source of input lag, especially on lower-end devices, even with pruning.
-   **Short Word Accuracy**: Like all path-based systems, this one likely struggles to differentiate short, common words (in, on, it, is, at) where the geometric path information is minimal.

---

## 3. Ways to Prune Early (How-To)

The current `SwipePruner` is a good start. Further improvements can be made:

-   **Implement Trie-Based Dictionary Search**:
    -   **Files**: `WordPredictor.java`, `DTWPredictor.java`
    -   **How**: Currently, the system appears to iterate through lists of words. This should be replaced with a **Trie** data structure. As the gesture path is analyzed and a sequence of probable keys is generated (e.g., `q-w-e-r`), this sequence can be used to traverse the Trie, instantly filtering the dictionary to only words with that prefix. This is dramatically more efficient than list iteration.

---

## 4. Optimizations to Reduce Input Lag

-   **Fully Implement the ML Model**:
    -   **Files**: `SwipeTypingEngine.java`, `Keyboard2.java`
    -   **How**: The single biggest optimization is to **replace the entire DTW/Sequence/Scorer pipeline with the pre-trained ML model**. The `swipe_model_android.pte` file suggests it's an Executorch model. The app should load this model and use it for inference. This would unify the architecture and likely provide a significant performance and accuracy boost.
-   **Quantize the ML Model**:
    -   **Files**: `models/`, `ml_training/`
    -   **How**: If it's not already, the neural network model should be quantized (e.g., to INT8). This dramatically reduces model size and speeds up inference on mobile CPUs, significantly reducing lag.
-   **Enable Hardware Acceleration**:
    -   **Files**: `SwipeTypingEngine.java` (or a new ML inference class)
    -   **How**: When running inference with the ML model, use the Android NNAPI delegate. This allows the model to run on specialized hardware (GPU/DSP) if available, which is much faster and more power-efficient than running on the CPU.
-   **Ensure Asynchronous Execution**: The current system correctly uses an `AsyncPredictionHandler` to move prediction off the UI thread. This practice must be maintained when switching to an ML-based model.
