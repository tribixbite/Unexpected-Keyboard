#!/usr/bin/env python3
"""
Deduplicate and fix contraction handling

Issues to fix:
1. 11 spurious 'ss' words (jesuss, jamess, etc.) should be removed
2. 1,104 possessives should map to base word in contraction_pairings (not non_paired)
3. Only 74 REAL contractions should be in non_paired
4. 6 possessives without base words should be removed entirely

Strategy:
- Possessives (word's) → map to base word in pairings (e.g., "obama's" shows as ["obama", "obama's"])
- Real contractions (don't, can't, we'll) → stay in non_paired (e.g., "dont" displays as "don't")
"""

import json
import sys

def categorize_contractions(original_contractions, current_dict):
    """Categorize all contractions properly"""

    # Define real contraction patterns (not possessives)
    real_contraction_indicators = {
        "n't",  # negatives
        "'m",   # am
        "'re",  # are
        "'ve",  # have
        "'ll",  # will
        "'d"    # would/had
    }

    # Common 's contractions that are is/has (not possessive)
    is_has_contractions = {
        "it's", "that's", "what's", "he's", "she's", "there's", "here's",
        "let's", "who's", "where's", "how's", "when's"
    }

    real_contractions = {}
    possessives_with_base = {}
    possessives_without_base = {}
    spurious_ss_words = []

    for word, freq in original_contractions.items():
        without_apos = word.replace("'", "")

        # Check if it's a real contraction
        is_real_contraction = False
        for indicator in real_contraction_indicators:
            if word.endswith(indicator):
                is_real_contraction = True
                break

        if word in is_has_contractions:
            is_real_contraction = True

        if is_real_contraction:
            # Real contraction - goes to non_paired
            real_contractions[without_apos] = word
        else:
            # Possessive (word's)
            base_word = word[:-2]  # Remove 's

            if base_word in current_dict:
                # Base word exists - add to pairings
                if base_word not in possessives_with_base:
                    possessives_with_base[base_word] = []

                possessives_with_base[base_word].append({
                    'contraction': word,
                    'frequency': freq
                })

                # Check for spurious 'ss' words
                if base_word.endswith('s') and without_apos in current_dict:
                    # This creates a double-s problem!
                    # e.g., "jesus's" → "jesuss" (spurious) when "jesus" (base) exists
                    spurious_ss_words.append(without_apos)
            else:
                # Base word doesn't exist - orphaned possessive
                possessives_without_base[without_apos] = word

    return {
        'real_contractions': real_contractions,
        'possessives_with_base': possessives_with_base,
        'possessives_without_base': possessives_without_base,
        'spurious_ss_words': spurious_ss_words
    }

def main():
    print("=== CONTRACTION DEDUPLICATION AND FIX ===\n")

    # Load original contractions
    with open('docs/dictionaries/contractions_all_original.json', 'r') as f:
        original_contractions = json.load(f)

    # Load current dictionary
    with open('assets/dictionaries/en_enhanced.json', 'r') as f:
        current_dict = json.load(f)

    print(f"Loaded {len(original_contractions)} original contractions")
    print(f"Loaded {len(current_dict)} words from current dictionary\n")

    # Categorize everything
    categories = categorize_contractions(original_contractions, current_dict)

    print("=== CATEGORIZATION RESULTS ===\n")
    print(f"Real contractions (non-paired): {len(categories['real_contractions'])}")
    print(f"Possessives with base word (pairings): {len(categories['possessives_with_base'])}")
    print(f"Possessives without base (to remove): {len(categories['possessives_without_base'])}")
    print(f"Spurious 'ss' words (to remove): {len(categories['spurious_ss_words'])}")

    # Show spurious ss words
    print("\nSpurious 'ss' words to be removed:")
    for word in sorted(categories['spurious_ss_words']):
        freq = current_dict[word]
        print(f"  {word:20s} (freq {freq})")

    # Show orphaned possessives
    print("\nOrphaned possessives to be removed:")
    for without_apos, with_apos in sorted(categories['possessives_without_base'].items()):
        if without_apos in current_dict:
            freq = current_dict[without_apos]
            print(f"  {with_apos:25s} → {without_apos:20s} (freq {freq}, base word missing)")

    # Clean dictionary
    print("\n=== CLEANING DICTIONARY ===\n")

    words_to_remove = set(categories['spurious_ss_words']) | set(categories['possessives_without_base'].keys())
    original_count = len(current_dict)

    for word in words_to_remove:
        if word in current_dict:
            del current_dict[word]

    print(f"Removed {original_count - len(current_dict)} words from dictionary")
    print(f"Dictionary now has {len(current_dict)} words")

    # Build new contraction_pairings.json
    print("\n=== BUILDING CONTRACTION PAIRINGS ===\n")

    # Load existing pairings (non-possessive ones)
    with open('assets/dictionaries/contraction_pairings.json', 'r') as f:
        old_pairings = json.load(f)

    # Start with possessives
    new_pairings = categories['possessives_with_base'].copy()

    # Add existing non-possessive pairings
    for base, contractions in old_pairings.items():
        if base in new_pairings:
            # Merge
            for c in contractions:
                if c not in new_pairings[base]:
                    new_pairings[base].append(c)
        else:
            new_pairings[base] = contractions

    print(f"Total paired base words: {len(new_pairings)}")
    total_pairs = sum(len(v) for v in new_pairings.values())
    print(f"Total contraction variants: {total_pairs}")

    # Show top 20 pairings
    print("\nTop 20 pairings by frequency:")
    all_pairs = []
    for base, contractions in new_pairings.items():
        for c in contractions:
            freq = c['frequency'] if isinstance(c, dict) else current_dict.get(base, 0)
            all_pairs.append((base, c, freq))

    for base, c, freq in sorted(all_pairs, key=lambda x: x[2], reverse=True)[:20]:
        contraction_str = c['contraction'] if isinstance(c, dict) else c
        print(f"  {base:20s} → {contraction_str:25s} (freq {freq})")

    # Build new contractions_non_paired.json
    print("\n=== BUILDING NON-PAIRED CONTRACTIONS ===\n")

    new_non_paired = categories['real_contractions'].copy()

    print(f"Total non-paired contractions: {len(new_non_paired)}")
    print("\nAll non-paired contractions:")
    for without_apos, with_apos in sorted(new_non_paired.items()):
        print(f"  {without_apos:15s} → {with_apos:20s}")

    # Save everything
    print("\n=== SAVING FILES ===\n")

    # Save cleaned dictionary
    with open('assets/dictionaries/en_enhanced.json', 'w', encoding='utf-8') as f:
        json.dump(current_dict, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"✓ Saved cleaned dictionary: {len(current_dict)} words")

    # Save new pairings
    with open('assets/dictionaries/contraction_pairings.json', 'w', encoding='utf-8') as f:
        json.dump(new_pairings, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"✓ Saved contraction_pairings.json: {len(new_pairings)} base words, {total_pairs} variants")

    # Save new non-paired
    with open('assets/dictionaries/contractions_non_paired.json', 'w', encoding='utf-8') as f:
        json.dump(new_non_paired, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"✓ Saved contractions_non_paired.json: {len(new_non_paired)} contractions")

    # Copy to docs/
    with open('docs/dictionaries/contraction_pairings.json', 'w', encoding='utf-8') as f:
        json.dump(new_pairings, f, ensure_ascii=False, indent=2, sort_keys=True)

    with open('docs/dictionaries/contractions_non_paired.json', 'w', encoding='utf-8') as f:
        json.dump(new_non_paired, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("✓ Copied to docs/dictionaries/")

    print("\n=== SUMMARY ===\n")
    print(f"Dictionary: {len(current_dict)} words (removed {original_count - len(current_dict)})")
    print(f"Paired contractions: {len(new_pairings)} base words → {total_pairs} variants")
    print(f"Non-paired contractions: {len(new_non_paired)} real contractions")
    print("\nRun regenerate_txt_dictionary.py next to update en_enhanced.txt")

if __name__ == '__main__':
    main()
