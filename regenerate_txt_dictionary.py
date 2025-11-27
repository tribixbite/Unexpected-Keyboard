#!/usr/bin/env python3
"""
Regenerate en_enhanced.txt from modified JSON dictionary
Converts JSON format to simple text file (one word per line, sorted by frequency)
"""

import json
import sys

def regenerate_txt_from_json():
    """Convert modified JSON dictionary to text format for calibration"""

    # Load modified JSON dictionary
    with open('assets/dictionaries/en_enhanced.json', 'r', encoding='utf-8') as f:
        dictionary = json.load(f)

    print(f"Loaded {len(dictionary)} words from modified JSON dictionary")

    # Sort by frequency (descending) - highest frequency first
    sorted_words = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

    # Write to text file (one word per line)
    output_file = 'assets/dictionaries/en_enhanced.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, freq in sorted_words:
            f.write(f"{word}\n")

    print(f"Generated {output_file} with {len(sorted_words)} words")
    print(f"Frequency range: {sorted_words[0][1]} (highest) to {sorted_words[-1][1]} (lowest)")

    # Verify no apostrophes
    words_with_apostrophes = [word for word, freq in sorted_words if "'" in word]
    if words_with_apostrophes:
        print(f"\nWARNING: Found {len(words_with_apostrophes)} words with apostrophes:")
        for word in words_with_apostrophes[:10]:
            print(f"  - {word}")
    else:
        print("\n✓ Verified: No words with apostrophes in modified dictionary")

    # Show some examples
    print("\nFirst 20 words (highest frequency):")
    for word, freq in sorted_words[:20]:
        print(f"  {word} ({freq})")

if __name__ == '__main__':
    regenerate_txt_from_json()
