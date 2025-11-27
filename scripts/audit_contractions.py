#!/usr/bin/env python3
"""
Audit script for contraction_pairings.json

Separates true contractions from simple possessives to reduce bloat.

True contractions: irregular forms like "don't", "won't", "aren't"
Simple possessives: predictable forms like "cat's", "dog's", "aaron's"

Usage:
    python3 scripts/audit_contractions.py
"""

import json
import os
import sys

# Common pronouns and function words that form true contractions
TRUE_CONTRACTION_BASES = {
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'who', 'what', 'that', 'there', 'here',

    # Auxiliary verbs
    'is', 'am', 'are', 'was', 'were',
    'have', 'has', 'had',
    'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must',
    'do', 'does', 'did', 'need',

    # Negations
    'not', 'no', 'never',

    # Others
    'let', 'ought', 'dare'
}

# True irregular contractions (not predictable by simple rules)
IRREGULAR_CONTRACTIONS = {
    "won't",  # will not
    "shan't",  # shall not
    "can't",  # cannot
    "ain't",  # am not / is not (non-standard)
}

def is_true_contraction(word, contraction):
    """
    Determine if this is a true contraction or just a possessive.

    True contractions:
    - End with 't (negative), 'll (will), 'd (would/had), 're (are), 've (have), 'm (am)
    - Base word is a pronoun or function word
    - Or are irregular forms like "won't"

    Simple possessives:
    - End with 's (possessive)
    - Base word is a noun
    """
    word_lower = word.lower()
    contraction_lower = contraction.lower()

    # Simple possessive: word + 's
    if contraction_lower == word_lower + "'s" or contraction_lower == word_lower + "s'":
        # Only true contraction if base is a pronoun/function word
        return word_lower in TRUE_CONTRACTION_BASES

    # Check for irregular contractions
    if contraction_lower in IRREGULAR_CONTRACTIONS:
        return True

    # True contraction patterns
    true_patterns = ["'t", "'ll", "'d", "'re", "'ve", "'m"]
    for pattern in true_patterns:
        if pattern in contraction_lower and not contraction_lower.endswith("'s"):
            # Must be based on a pronoun/function word
            return word_lower in TRUE_CONTRACTION_BASES

    return False

def audit_contractions():
    """Main audit function."""

    input_file = "assets/dictionaries/contraction_pairings.json"
    output_clean = "assets/dictionaries/contraction_pairings_cleaned.json"
    output_audit = "assets/dictionaries/possessives_audit.txt"

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    true_contractions = {}
    simple_possessives = []

    total_entries = 0
    total_contractions = 0

    for word, expansions in data.items():
        for expansion in expansions:
            total_entries += 1
            contraction = expansion['contraction']
            frequency = expansion.get('frequency', 128)

            if is_true_contraction(word, contraction):
                # Keep as true contraction
                if word not in true_contractions:
                    true_contractions[word] = []
                true_contractions[word].append(expansion)
                total_contractions += 1
            else:
                # Mark as simple possessive
                simple_possessives.append({
                    'word': word,
                    'possessive': contraction,
                    'frequency': frequency
                })

    # Write cleaned contraction file (only true contractions)
    print(f"\nWriting {output_clean}...")
    with open(output_clean, 'w') as f:
        json.dump(true_contractions, f, indent=2, sort_keys=True)

    # Write audit log (all removed possessives)
    print(f"Writing {output_audit}...")
    with open(output_audit, 'w') as f:
        f.write("# Possessives Audit Log\n")
        f.write(f"# Removed {len(simple_possessives)} simple possessive entries\n")
        f.write(f"# These will be generated dynamically by rules instead\n\n")

        for entry in sorted(simple_possessives, key=lambda x: x['word']):
            f.write(f"{entry['word']:20s} -> {entry['possessive']:25s} (freq: {entry['frequency']})\n")

    # Print summary
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    print(f"Total entries processed:       {total_entries}")
    print(f"True contractions (kept):      {total_contractions}")
    print(f"Simple possessives (removed):  {len(simple_possessives)}")
    print(f"Reduction:                     {len(simple_possessives)/total_entries*100:.1f}%")
    print("\nFiles created:")
    print(f"  - {output_clean}")
    print(f"  - {output_audit}")

    # Show sample true contractions
    print("\nSample TRUE contractions (kept):")
    for word in sorted(true_contractions.keys())[:15]:
        for exp in true_contractions[word]:
            print(f"  {word:10s} -> {exp['contraction']}")

    # Show sample possessives
    print("\nSample simple POSSESSIVES (removed, will use rules):")
    for entry in simple_possessives[:15]:
        print(f"  {entry['word']:10s} -> {entry['possessive']}")

    print("\n" + "="*60)

    # File size comparison
    import os
    original_size = os.path.getsize(input_file)
    cleaned_size = os.path.getsize(output_clean)
    print(f"\nFile size reduction:")
    print(f"  Original: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    print(f"  Cleaned:  {cleaned_size:,} bytes ({cleaned_size/1024:.1f} KB)")
    print(f"  Savings:  {original_size-cleaned_size:,} bytes ({(original_size-cleaned_size)/1024:.1f} KB)")
    print(f"  Reduction: {(original_size-cleaned_size)/original_size*100:.1f}%")

    return True

if __name__ == "__main__":
    try:
        success = audit_contractions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
