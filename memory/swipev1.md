# Breakdown: The "Swipe Calibration" Screen

This document details the functionality of the "Swipe Calibration" screen. Contrary to a typical user-facing calibration tool, this screen is a comprehensive **developer utility** for data collection, debugging, and algorithm analysis.

Its primary purpose is not to tune the keyboard for the user, but to generate high-quality swipe data for training machine learning models and to provide a detailed analysis of how the underlying prediction engines perform on any given swipe.

**Key File:** `srcs/juloo/keyboard2/SwipeCalibrationActivity.java`

---

## Core Functionalities

1.  **Swipe Data Collection**: The user is prompted to swipe a series of words. Each swipe (a path of x,y coordinates and timestamps) is captured and stored in a local SQLite database (`swipe_ml_data.db`) as a `SwipeMLData` object. This data is used for offline model training.

2.  **Prediction Analysis**: After each swipe, the activity calls the app's core prediction engines to see how they would rank candidates for that swipe. This is for analysis, not for user-facing correction.

3.  **Debug Visualization**: The screen displays the user's exact swipe path overlaid on the keyboard and provides a text-based breakdown of the scores from the various algorithms.

---

## How Prediction Ranking Scores are Computed

The ranking score displayed in this activity is a direct look into the production prediction engine. The primary algorithm used for this analysis appears to be the `KeyboardSwipeRecognizer`, which uses a Bayesian-inspired framework:

**P(word | swipe) ∝ P(swipe | word) × P(word)**

-   `P(word | swipe)`: The final probability (and thus the rank) of a `word` given the `swipe`.
-   `P(swipe | word)`: The **Likelihood**. How likely is this `swipe` path if the user intended to type this `word`? This is the geometric score.
-   `P(word)`: The **Prior**. How likely is the `word` in general? This is the language model or frequency score.

The final score is a combination of several weighted components:

1.  **Proximity Score (Geometric Match)**: This is the core geometric calculation. It compares the user's swipe path to the ideal path for a candidate word (the "template"). It penalizes for:
    *   The distance between the swipe path and the template path.
    *   Missing required letters from the word.
    *   Including extra letters not in the word.
    *   Incorrect ordering of the letters.

2.  **Start Point Score**: A bonus is applied if the swipe starts very close to the first letter of the candidate word. This gives strong weight to the user's initial intention.

3.  **Language Model Score (Frequency)**: The score is boosted based on the word's general frequency in the English language. Common words are considered more likely.

### Example Calculation

Imagine a user swipes a path for the word **"hello"**. The system analyzes several candidates, including "hello" and "jello".

1.  **Candidate: "hello"**
    *   **Proximity Score**: The user's path is very close to the ideal path for "hello". It passes over H-E-L-L-O in the correct order. **Score: 0.95**
    *   **Start Point Score**: The swipe started very close to the 'H' key. **Score: 0.98**
    *   **Language Score**: "hello" is a very common word. **Score: 0.90**
    *   **Final Score (conceptual)**: `(0.95 * W_prox) + (0.98 * W_start) + (0.90 * W_lang) = High Score`

2.  **Candidate: "jello"**
    *   **Proximity Score**: The path is somewhat close, but the start is off.
        *   It will be heavily penalized for missing the 'J' at the beginning.
        *   It will be penalized for having an extra 'H' at the start.
        *   **Score: 0.40**
    *   **Start Point Score**: The swipe started near 'H', not 'J'. **Score: 0.10**
    *   **Language Score**: "jello" is a reasonably common word. **Score: 0.65**
    *   **Final Score (conceptual)**: `(0.40 * W_prox) + (0.10 * W_start) + (0.65 * W_lang) = Low Score`

### Conclusion

The calibration screen does not compute its own scores. Instead, it acts as a sophisticated test harness that invokes the real prediction engine (`KeyboardSwipeRecognizer`) and displays the resulting scores and debug information, allowing developers to analyze and improve the core swipe-typing algorithms.
