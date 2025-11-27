#!/usr/bin/env python3
"""
Process dictionary to handle contractions properly.

Steps:
1. Load original dictionary
2. Find all words with apostrophes
3. Categorize into paired/non-paired
4. Generate reference files
5. Create modified dictionary
"""

import json
import sys

# Common contractions to ensure we have (from user's list)
COMMON_CONTRACTIONS = [
    # To be verbs
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
    # To have verbs
    "i've", "you've", "we've", "they've",
    # Auxiliary verbs - will
    "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll",
    # Auxiliary verbs - would
    "i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd",
    # Negatives
    "can't", "don't", "doesn't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "couldn't", "shouldn't", "wouldn't",
    "hasn't", "haven't",
    # Question words
    "what's", "what'd", "when's", "where's", "who's", "who'd",
    "how's", "how'd",
    # Other common
    "that's", "there's", "here's", "let's"
]

def load_dictionary(path):
    """Load JSON dictionary."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    """Save JSON with proper formatting."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_list(items, path):
    """Save list of strings, one per line."""
    with open(path, 'w') as f:
        for item in sorted(items):
            f.write(f"{item}\n")

def process_contractions(dict_path):
    """Process dictionary and categorize contractions."""

    print("Loading dictionary...")
    original_dict = load_dictionary(dict_path)

    # Find all words with apostrophes
    words_with_apostrophes = {word: freq for word, freq in original_dict.items() if "'" in word}
    print(f"Found {len(words_with_apostrophes)} words with apostrophes")

    # Categorize
    contraction_pairings = {}  # {base_word: [contraction1, contraction2, ...]}
    non_paired_contractions = {}  # {without_apostrophe: with_apostrophe}

    for contraction, freq in words_with_apostrophes.items():
        # Remove apostrophe
        without_apostrophe = contraction.replace("'", "")

        # Check if the non-apostrophe version exists in dictionary
        if without_apostrophe in original_dict:
            # PAIRED - the base word exists
            print(f"  PAIRED: '{contraction}' → '{without_apostrophe}' (exists)")

            if without_apostrophe not in contraction_pairings:
                contraction_pairings[without_apostrophe] = []
            contraction_pairings[without_apostrophe].append({
                'contraction': contraction,
                'frequency': freq
            })
        else:
            # NON-PAIRED - base word doesn't exist
            print(f"  NON-PAIRED: '{contraction}' → '{without_apostrophe}' (new)")
            non_paired_contractions[without_apostrophe] = contraction

    # Create modified dictionary
    modified_dict = {}

    # Copy all non-apostrophe words
    for word, freq in original_dict.items():
        if "'" not in word:
            modified_dict[word] = freq

    # Add non-paired contractions (without apostrophes)
    for without_apostrophe, with_apostrophe in non_paired_contractions.items():
        # Use the original frequency
        modified_dict[without_apostrophe] = words_with_apostrophes[with_apostrophe]
        print(f"  ADDED to dict: '{without_apostrophe}' (was '{with_apostrophe}')")

    # For paired contractions, the base word already exists, so we don't add anything
    # We just track the pairing for later use

    print(f"\n=== Summary ===")
    print(f"Original dictionary: {len(original_dict)} words")
    print(f"Words with apostrophes: {len(words_with_apostrophes)}")
    print(f"Contraction pairings: {len(contraction_pairings)} base words with {sum(len(v) for v in contraction_pairings.values())} contractions")
    print(f"Non-paired contractions: {len(non_paired_contractions)}")
    print(f"Modified dictionary: {len(modified_dict)} words")

    return {
        'original_contractions': words_with_apostrophes,
        'contraction_pairings': contraction_pairings,
        'non_paired_contractions': non_paired_contractions,
        'modified_dictionary': modified_dict
    }

def ensure_common_contractions(data):
    """Ensure all common contractions are included."""
    print("\n=== Checking common contractions ===")

    all_contractions = set()

    # Collect all contractions we have
    for base, pairings in data['contraction_pairings'].items():
        for pairing in pairings:
            all_contractions.add(pairing['contraction'])

    for with_apos in data['non_paired_contractions'].values():
        all_contractions.add(with_apos)

    # Check which common ones are missing
    missing = []
    for common in COMMON_CONTRACTIONS:
        if common not in all_contractions:
            missing.append(common)

    if missing:
        print(f"⚠️  Missing {len(missing)} common contractions:")
        for m in missing:
            print(f"    - {m}")
    else:
        print(f"✅ All {len(COMMON_CONTRACTIONS)} common contractions present!")

    return missing

if __name__ == "__main__":
    dict_path = "assets/dictionaries/en_enhanced.json"
    output_dir = "docs/dictionaries"

    # Process
    data = process_contractions(dict_path)

    # Check coverage
    missing = ensure_common_contractions(data)

    # Save reference files
    print("\n=== Saving reference files ===")

    # 1. All original contractions with frequencies
    save_json(data['original_contractions'], f"{output_dir}/contractions_all_original.json")
    print(f"✅ Saved: {output_dir}/contractions_all_original.json")

    # 2. Contraction pairings (base word → contractions)
    save_json(data['contraction_pairings'], f"{output_dir}/contraction_pairings.json")
    print(f"✅ Saved: {output_dir}/contraction_pairings.json")

    # 3. Non-paired contractions mapping (without → with apostrophe)
    save_json(data['non_paired_contractions'], f"{output_dir}/contractions_non_paired.json")
    print(f"✅ Saved: {output_dir}/contractions_non_paired.json")

    # 4. Simple list for easy reading
    paired_list = []
    for base, pairings in data['contraction_pairings'].items():
        for pairing in pairings:
            paired_list.append(f"{base} ↔ {pairing['contraction']}")
    save_list(paired_list, f"{output_dir}/contraction_pairings_list.txt")
    print(f"✅ Saved: {output_dir}/contraction_pairings_list.txt")

    non_paired_list = [f"{k} → {v}" for k, v in data['non_paired_contractions'].items()]
    save_list(non_paired_list, f"{output_dir}/contractions_non_paired_list.txt")
    print(f"✅ Saved: {output_dir}/contractions_non_paired_list.txt")

    # 5. Modified dictionary (NO apostrophes)
    save_json(data['modified_dictionary'], "assets/dictionaries/en_enhanced_modified.json")
    print(f"✅ Saved: assets/dictionaries/en_enhanced_modified.json")

    print("\n=== ✅ Processing complete! ===")
    print(f"\nNext steps:")
    print(f"1. Review the reference files in {output_dir}/")
    print(f"2. Backup original: assets/dictionaries/en_enhanced.json")
    print(f"3. Replace with modified: assets/dictionaries/en_enhanced_modified.json")
    print(f"4. Update Java code to use pairing/non-paired lists")
