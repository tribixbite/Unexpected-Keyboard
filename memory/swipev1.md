# Swipe Prediction Scoring for Calibration Screen

This document provides a detailed breakdown of how the prediction ranking scores are computed within the swipe calibration screen. The goal of the scoring system is to rank a list of candidate words based on how well they match a user's swipe gesture.

The final score for each candidate word is a weighted combination of two primary scores:
1.  **Geometric Score (`S_geom`):** Measures the similarity between the user's swipe path and the ideal path for the candidate word.
2.  **Lexical Score (`S_lex`):** Measures the likelihood of the candidate word appearing in the language (based on frequency).

---

## 1. Gesture Path Preprocessing

Before scoring, the raw user input (a sequence of x, y coordinates) is normalized to account for variations in speed and keyboard scale. This creates a standardized path that can be compared against ideal word paths.

- **Input:** Raw path `P_raw = [(x1, y1), (x2, y2), ...]`
- **Output:** Normalized path `P_norm`

---

## 2. Candidate Word Generation

The system generates a list of potential candidate words from the dictionary. This is done by identifying all words whose constituent characters lie near the user's swipe path.

---

## 3. Scoring and Ranking Algorithm

For each candidate word, the system calculates a final score and ranks them.

### a. Geometric Score (`S_geom`)

This score quantifies how well the shape of the user's normalized swipe path (`P_norm`) matches the ideal, pre-calculated path for a candidate word (`P_ideal`). The ideal path is the straight-line path connecting the centers of the keys for that word.

The similarity is calculated using a path similarity algorithm like Dynamic Time Warping (DTW), which finds the optimal alignment between two sequences. The result is a score between 0.0 (no match) and 1.0 (perfect match).

**Formula:** `S_geom = 1 - (DTW_distance(P_norm, P_ideal) / Max_Possible_Distance)`

### b. Lexical Score (`S_lex`)

This score is determined by the frequency of the candidate word in the active dictionary. More common words are considered more likely and receive a higher score. The frequency is normalized to a value between 0.0 and 1.0.

**Formula:** `S_lex = NormalizedWordFrequency(candidate_word)`

### c. Final Score (`S_final`)

The final score is a weighted average of the Geometric and Lexical scores. The weights are adjustable parameters, but typical values emphasize the geometric match slightly more than the lexical probability.

- `W_geom`: Weight for the geometric score (e.g., 0.6)
- `W_lex`: Weight for the lexical score (e.g., 0.4)

**Formula:** `S_final = (W_geom * S_geom) + (W_lex * S_lex)`

---

## Example Calculation

Let's assume the user swipes a path that is close to the word "good".

**User Swipe:** A slightly curved path near the keys G, O, D.
**Candidate Words:** The system identifies "good", "god", and "food" as potential matches.
**Weights:** `W_geom = 0.6`, `W_lex = 0.4`

### Scoring:

1.  **Candidate: "good"**
    - The user's path is very similar to the ideal path for "good".
      - `S_geom` = **0.95**
    - "good" is a very common word.
      - `S_lex` = **0.90**
    - `S_final("good")` = (0.6 * 0.95) + (0.4 * 0.90) = 0.57 + 0.36 = **0.93**

2.  **Candidate: "god"**
    - The path is shorter and doesn't account for the double 'o'.
      - `S_geom` = **0.70**
    - "god" is a common word, but less so than "good".
      - `S_lex` = **0.75**
    - `S_final("god")` = (0.6 * 0.70) + (0.4 * 0.75) = 0.42 + 0.30 = **0.72**

3.  **Candidate: "food"**
    - The path starts closer to 'G' than 'F'.
      - `S_geom` = **0.60**
    - "food" is a very common word.
      - `S_lex` = **0.88**
    - `S_final("food")` = (0.6 * 0.60) + (0.4 * 0.88) = 0.36 + 0.352 = **0.712**

### Final Ranking

The candidates are sorted by their `S_final` score in descending order.

| Rank | Candidate | `S_geom` | `S_lex` | `S_final` |
|------|-----------|----------|---------|-----------|
| 1    | good      | 0.95     | 0.90    | **0.930** |
| 2    | god       | 0.70     | 0.75    | **0.720** |
| 3    | food      | 0.60     | 0.88    | **0.712** |

The calibration screen would display this ranked list, with "good" correctly identified as the top prediction.
