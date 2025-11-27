# Performance Todos v5: Optimizing the Contraction System

This document outlines an analysis of the contraction system and a plan to re-architect it for much greater efficiency and flexibility by replacing rote memorization with a rule-based approach.

## I. Summary of Findings

An analysis of the contraction data files (`contraction_pairings.json` and `contractions_non_paired.json`) has revealed a major architectural inefficiency.

*   `contractions_non_paired.json`: This file is clean and contains only true, irregular contractions (e.g., `dont` -> `don't`). It is functioning as intended.
*   `contraction_pairings.json`: This file is **massively bloated**. It contains thousands of entries that are not true contractions but simple, predictable possessives (e.g., `"aaron": [{"contraction": "aaron's"}]`, `"cat": [{"contraction": "cat's"}]`).

Storing thousands of simple possessives in a static list is highly inefficient. It inflates the size of the final `contractions.bin`, increases the memory required at runtime, and is inflexible—it can't handle the possessive of any word not in the list.

## II. Proposed Solution: Hybrid Rule-Based System

The current system should be replaced with a hybrid approach that uses rules for predictable possessives and a small list for true, irregular contractions.

### Todo 1 (High Priority): Audit and Clean Contraction Data

**Problem:** The `contraction_pairings.json` file is filled with thousands of unnecessary possessive entries.

**Solution:** Create a script to parse the existing file and separate the true contractions from the simple possessives.

**Action Items:**
1.  **Create an audit script** (e.g., `scripts/audit_contractions.py`).
2.  The script should read `contraction_pairings.json` and apply heuristics to classify each entry.
    *   **True Contraction:** A word that is a shortened form of another phrase (e.g., "it's" -> "it is", "he'll" -> "he will"). The base word ("it", "he") is a pronoun or common function word.
    *   **Simple Possessive:** A word where `'s` is added to a noun (e.g., `cat's`, `aaron's`, `company's`).
3.  The script should output two files:
    *   `contraction_pairings_cleaned.json`: Contains only the true, irregular contractions. This file will be very small.
    *   `possessives_audit.txt`: A log of all the simple possessives that were removed, for verification purposes.

---

### Todo 2 (High Priority): Update Build Process and Data

**Problem:** The build process currently uses the bloated JSON file to create `contractions.bin`.

**Solution:** Update the build process to use the new, cleaned JSON file.

**Action Items:**
1.  Modify the script that generates `contractions.bin` (e.g., `scripts/generate_binary_contractions.py`) to take `contraction_pairings_cleaned.json` as its input instead of the original file.
2.  Re-run the script to generate a new, much smaller `contractions.bin`.
3.  Commit both the new `contractions.bin` and the `contraction_pairings_cleaned.json` file to the repository, and remove or archive the original bloated JSON file.

**Benefit:** This will immediately reduce the app's binary size and memory usage related to contractions.

---

### Todo 3 (Medium Priority): Implement Rule-Based Possessive Generation

**Problem:** With the static list of possessives removed, the prediction engine can no longer suggest them. A dynamic, rule-based system is needed.

**Solution:** Modify the prediction engine to intelligently offer the `'s` possessive form for relevant word candidates.

**Action Items:**
1.  **Location:** The logic should be added within the prediction generation loop (e.g., in `WordPredictor.predictInternal` or a method it calls).
2.  **Implementation Strategy:**
    *   After a list of initial candidates is generated (e.g., from a swipe), iterate through them.
    *   For each candidate word (e.g., "dog", "house", "chris"), generate a corresponding possessive candidate ("dog's", "house's", "chris's").
        *   **Simple Rule:** For words not ending in 's', append `'s`.
        *   **Advanced Rule (Recommended):** For words ending in 's' (like "chris"), append only an apostrophe (`'`) or `'s` based on style guides (e.g., "chris's"). The simpler `'s` rule is acceptable for a first pass.
    *   Add these new possessive candidates to the list of candidates to be scored and ranked along with the original ones.
3.  **Scoring:** The scoring mechanism (`calculateUnifiedScore`) should naturally handle these new candidates. A word like "dog's" will have a similar frequency and contextual fit to "dog", so it should appear as a relevant suggestion when appropriate.

**Benefit:** The keyboard will now be able to predict the possessive form of **any word in the dictionary**, not just the thousands that were hard-coded, making the feature infinitely more flexible and useful. This comes at a very low computational cost and with a significant reduction in data size.
