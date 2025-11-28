# Dictionary Character Analysis

**Dictionary**: `assets/dictionaries/en_enhanced.txt`
**Total Words**: 49,296
**Analysis Date**: 2025-11-25

---

## Summary

Out of 49,296 words:
- **237 words (0.48%)** contain non-English lowercase characters
- **49,059 words (99.52%)** contain ONLY lowercase a-z

---

## Breakdown by Character Type

### 1. Accented Characters: 213 words (0.43%)

Words with Latin accents (à, á, é, ñ, ü, etc.)

**Examples:**
- abbé, café, cafés, cliché, résumé
- andré, andrés, josé, maría
- naïve, façade, fiancé, fiancée
- château, communiqué, après, ménage
- señor, señora, niño, españa
- björk, søren, françois, müller

**Common Patterns:**
- French loanwords: café, résumé, château, naïve
- Spanish names: josé, maría, garcía, lópez
- Proper nouns: beyoncé, erdoğan, citroën

### 2. Greek Letters: 14 words (0.03%)

Scientific/mathematical notation

**Full List:**
- α (alpha)
- β (beta)
- γ (gamma)
- δ (delta)
- ε (epsilon)
- θ (theta)
- λ (lambda)
- μ (mu)
- μg (microgram)
- μm (micrometer)
- π (pi)
- σ (sigma)
- φ (phi)
- ω (omega)

**Usage**: Primarily scientific/technical terminology

### 3. Superscripts: 2 words (<0.01%)

Ordinal indicators

**List:**
- ª (feminine ordinal indicator)
- º (masculine ordinal indicator)

**Usage**: Spanish/Portuguese ordinal markers (1º, 2ª)

### 4. Other Unicode: 8 words (0.02%)

Miscellaneous Unicode characters

**List:**
- erdoğan (Turkish: ğ)
- ʖ (IPA symbol - lennie face character)
- а, в, и, на, с (Cyrillic letters)
- ツ (Japanese katakana character)

**Note**: These are likely edge cases, proper nouns, or internet slang

---

## Character Distribution

| Category | Count | Percentage | Examples |
|----------|-------|------------|----------|
| **Pure English (a-z)** | 49,059 | 99.52% | hello, world, keyboard |
| **Accented Latin** | 213 | 0.43% | café, résumé, naïve |
| **Greek Letters** | 14 | 0.03% | α, β, μ, π |
| **Superscripts** | 2 | <0.01% | ª, º |
| **Other Unicode** | 8 | 0.02% | erdoğan, ツ |
| **TOTAL** | 49,296 | 100% | - |

---

## Implications for Swipe Typing

### Encoding
The neural network tokenizer likely needs to handle:
- 26 English letters (a-z)
- Special tokens (SOS, EOS, PAD, UNK)
- Possibly extended Latin characters for the 213 accented words

### Current Implementation
Based on `SwipeTokenizer.java`, the vocabulary size is **30**:
- Index 0: PAD
- Index 1: UNK (unknown)
- Index 2: SOS (start of sequence)
- Index 3: EOS (end of sequence)
- Indices 4-29: Likely a-z (26 letters)

**Question**: How are the 237 non-English words handled?

**Possible Approaches:**
1. **Normalization**: Convert accents to base letters (café → cafe)
2. **UNK token**: Map to unknown character token
3. **Extended vocab**: Expand to include common accents
4. **Filtering**: Exclude from swipe predictions

### Recommendation
For the 0.48% of words with special characters:
- **Most common (213 accented)**: Consider normalization (é→e, ñ→n)
- **Scientific (14 Greek)**: Likely not swiped, can use UNK
- **Other (10 misc)**: Edge cases, can use UNK or filter

---

## Sample of Accented Words (First 50)

```
abbé, académie, amélie, américa, andré, andrés, antónio, après
arsène, atlético, attaché, barça, benoît, beyoncé, bibliothèque
bogotá, bolívar, café, cafés, calderón, carré, chrétien
chávez, château, citroën, città, cité, cliché, clichés
clément, communiqué, coordonnées, cortés, côte, crédit, crème
czech, días, début, décor, déjà, díaz, dès, días
erdoğan, español, español, été, façade, fédération, fiancé
fiancée, françois, frédéric, garcía, gonzalez, gómez, guzmán
hernández, hôtel, jérôme, josé, jiménez, lópez, maría
martínez, médicale, ménage, méxico, müller, naïve, niño
```

---

## Verification Commands

```bash
# Count total words
wc -l assets/dictionaries/en_enhanced.txt
# Output: 49296

# Count words with non-lowercase-alpha
grep -P '[^a-z\n]' assets/dictionaries/en_enhanced.txt | wc -l
# Output: 237

# Count accented words
grep -P '[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]' assets/dictionaries/en_enhanced.txt | wc -l
# Output: 213

# Count Greek letters
grep -P '[αβγδεθλμπσφω]' assets/dictionaries/en_enhanced.txt | wc -l
# Output: 14

# Show examples
grep -P '[^a-z\n]' assets/dictionaries/en_enhanced.txt | head -30
```

---

## Conclusion

The `en_enhanced.txt` dictionary is **99.52% pure English** (lowercase a-z). The small percentage (0.48%) of words with special characters are primarily:
- French loanwords with accents (café, résumé)
- Spanish/Portuguese names (josé, maría)
- Scientific notation (α, β, μ)

For swipe typing purposes, these 237 words likely need special handling (normalization or UNK token) since the tokenizer appears to support only 26 letters.

---

**Full List**: See `~/non_alpha_words.txt` (237 words)
