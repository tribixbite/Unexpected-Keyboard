# CGR Probability Calculation: A Deep Dive

This document provides a full breakdown of how the swipe prediction probabilities are calculated. The system is based on the `KeyboardSwipeRecognizer.java` file and uses a Bayesian framework to determine the most likely word for a given swipe.

The final probability (`prob`) seen in the logs is the result of **multiplying several independent scores together**. This is why the final values are often very small. The keyboard ranks candidates based on which one has the highest score, even if that score is a tiny fraction.

### The Core Equation

The final score is calculated as follows:

`Total Score = (Proximity Score × Sequence Score × Start Point Score) × Language Model Score`

-   The first three components combined represent the **Likelihood**: `P(swipe | word)` or "How likely is this swipe shape, given this specific word?"
-   The last component is the **Prior**: `P(word)` or "How common is this word in general?"

---

### Component 1: Proximity Score

-   **File:** `KeyboardSwipeRecognizer.java`
-   **Method:** `calculateProximityScore()`
-   **Purpose:** To measure the raw geometric similarity between the user's swipe path and the ideal template path for the word.

**How it Works:**
1.  It gets the ideal path (template) for the candidate word.
2.  It iterates through the points of the user's swipe.
3.  For each point, it calculates the physical distance to the corresponding point on the ideal template.
4.  This distance is converted into a score using an **exponential decay function**: `score = Math.exp(-distance / keyZoneRadius)`. This means:
    *   A perfect match (distance = 0) results in a score of 1.0.
    *   As the path deviates, the score rapidly gets smaller, approaching 0.
5.  The final score is the average of these individual point scores across the entire path.

---

### Component 2: Sequence Score

-   **File:** `KeyboardSwipeRecognizer.java`
-   **Method:** `calculateSequenceScore()`
-   **Purpose:** To apply severe penalties if the letters detected under the swipe path do not match the letters of the candidate word.

**How it Works:**
This score starts at `1.0` and is multiplied by crippling penalties for mismatches:

1.  **Missing Key Penalty**: For each letter in the candidate word that was **not** detected along the swipe path, the score is multiplied by `Math.exp(-10.0)`, which is `0.000045`. Missing even one required letter effectively gives the word a zero score.
2.  **Extra Key Penalty**: For each letter detected on the path that is **not** in the candidate word, the score is multiplied by `Math.exp(-2.0)`, which is `~0.135`.
3.  **Order Penalty**: If the detected letters appear in the wrong order compared to the candidate word, the score is multiplied by `Math.exp(-5.0)`, which is `~0.0067`.

**This is the most common reason for a `prob` of `0.000000`**. As seen with your "ESTATE" example, if the recognizer only confidently detects "e", "s", and "t", the word "estate" is penalized for missing the "a" and the second "e", making its `sequenceScore` (and thus its `totalScore`) effectively zero.

---

### Component 3: Start Point Score

-   **File:** `KeyboardSwipeRecognizer.java`
-   **Method:** `calculateStartPointScore()`
-   **Purpose:** To heavily reward swipes that begin near the first letter of the intended word.

**How it Works:**
1.  It measures the distance from the swipe's starting point to the center of the first letter's key.
2.  It converts this distance to a score using the same exponential decay function: `score = Math.exp(-startDistance / keyZoneRadius)`.
3.  Crucially, it **amplifies** this score by raising it to the power of `startPointWeight` (which defaults to 3.0). A score of `0.9` becomes `0.9^3 = 0.729`, while a score of `0.5` becomes `0.5^3 = 0.125`. This makes starting in the right place extremely important.

---

### Component 4: Language Model Score

-   **File:** `KeyboardSwipeRecognizer.java`
-   **Method:** `calculateLanguageModelScore()`
-   **Purpose:** To give a boost to words that are common or contextually likely.

**How it Works:**
This score is a combination of the word's general frequency, its likelihood given the previous word (bigram model), and a multiplier based on the user's personal typing habits.

### Conclusion: Why the Probabilities are So Low

The final `prob` is the result of multiplying four numbers that are almost always less than 1.0. Even if the geometry and start point are good (e.g., scores of 0.9 and 0.8), and the sequence is perfect (score of 1.0), the final score is still at the mercy of the language model's probability for that word, which can be very small for less common words.

For your "OWNER" example, the `prob` of `0.000041` is the result of multiplying several of these fractional scores together. It was ranked #1 simply because the scores for all other competing words were even smaller.