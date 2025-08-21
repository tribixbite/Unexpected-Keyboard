package juloo.keyboard2;

import android.content.Context;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Word prediction engine that matches swipe patterns to dictionary words
 */
public class WordPredictor
{
  private final Map<String, Integer> _dictionary;
  private final Map<Character, List<Character>> _adjacentKeys;
  private static final int MAX_PREDICTIONS = 5;
  private static final int MAX_EDIT_DISTANCE = 2;
  
  public WordPredictor()
  {
    _dictionary = new HashMap<>();
    _adjacentKeys = buildAdjacentKeysMap();
  }
  
  /**
   * Load dictionary from assets
   */
  public void loadDictionary(Context context, String language)
  {
    _dictionary.clear();
    // Try enhanced dictionary first, fall back to basic
    String filename = "dictionaries/" + language + "_enhanced.txt";
    String fallbackFilename = "dictionaries/" + language + ".txt";
    
    boolean dictionaryLoaded = false;
    
    // Only use enhanced dictionary, no fallback to basic
    try
    {
      BufferedReader reader = new BufferedReader(
        new InputStreamReader(context.getAssets().open(filename)));
      String line;
      while ((line = reader.readLine()) != null)
      {
        String[] parts = line.trim().split("\t");
        if (parts.length >= 1)
        {
          String word = parts[0].toLowerCase();
          int frequency = parts.length > 1 ? Integer.parseInt(parts[1]) : 1000;
          _dictionary.put(word, frequency);
        }
      }
      reader.close();
      dictionaryLoaded = true;
      android.util.Log.d("WordPredictor", "Loaded enhanced dictionary: " + filename + " with " + _dictionary.size() + " words");
    }
    catch (IOException e)
    {
      android.util.Log.e("WordPredictor", "Failed to load enhanced dictionary: " + filename + ", error: " + e.getMessage());
      // Don't fall back to basic dictionary - keep dictionary empty if enhanced not found
    }
  }
  
  /**
   * Reset the predictor state - called after space/punctuation
   */
  public void reset()
  {
    // This method will be called from Keyboard2 to reset state
    // Dictionary remains loaded, just clears any internal state if needed
    android.util.Log.d("WordPredictor", "Predictor reset");
  }
  
  /**
   * Build a map of adjacent keys on QWERTY keyboard
   */
  private Map<Character, List<Character>> buildAdjacentKeysMap()
  {
    Map<Character, List<Character>> adjacent = new HashMap<>();
    
    // QWERTY layout adjacency - simplified
    String[] rows = {
      "qwertyuiop",
      "asdfghjkl",
      "zxcvbnm"
    };
    
    for (int r = 0; r < rows.length; r++)
    {
      String row = rows[r];
      for (int c = 0; c < row.length(); c++)
      {
        char ch = row.charAt(c);
        List<Character> neighbors = new ArrayList<>();
        
        // Same row neighbors
        if (c > 0) neighbors.add(row.charAt(c - 1));
        if (c < row.length() - 1) neighbors.add(row.charAt(c + 1));
        
        // Adjacent row neighbors
        if (r > 0)
        {
          String prevRow = rows[r - 1];
          int offset = (r == 2) ? -1 : 0; // Adjust for keyboard stagger
          for (int i = Math.max(0, c + offset); i < Math.min(prevRow.length(), c + offset + 2); i++)
            neighbors.add(prevRow.charAt(i));
        }
        if (r < rows.length - 1)
        {
          String nextRow = rows[r + 1];
          int offset = (r == 1) ? 1 : 0; // Adjust for keyboard stagger
          for (int i = Math.max(0, c - offset); i < Math.min(nextRow.length(), c - offset + 2); i++)
            neighbors.add(nextRow.charAt(i));
        }
        
        adjacent.put(ch, neighbors);
      }
    }
    
    return adjacent;
  }
  
  /**
   * Predict words based on the sequence of touched keys
   */
  public List<String> predictWords(String keySequence)
  {
    if (keySequence == null || keySequence.isEmpty())
      return new ArrayList<>();
      
    List<WordCandidate> candidates = new ArrayList<>();
    String lowerSequence = keySequence.toLowerCase();
    
    android.util.Log.d("WordPredictor", "Predicting for: " + lowerSequence + " (len=" + lowerSequence.length() + ")");
    
    // Check if this is likely a swipe sequence (too many characters)
    boolean isSwipeSequence = lowerSequence.length() > 12;
    
    // Find all words that could match the key sequence
    for (Map.Entry<String, Integer> entry : _dictionary.entrySet())
    {
      String word = entry.getKey();
      int frequency = entry.getValue();
      
      // For swipe sequences, use special matching
      if (isSwipeSequence)
      {
        // For long sequences, prioritize first/last character matches
        if (word.length() > 0 && lowerSequence.length() > 0)
        {
          char firstChar = word.charAt(0);
          char lastChar = word.charAt(word.length() - 1);
          char seqFirst = lowerSequence.charAt(0);
          char seqLast = lowerSequence.charAt(lowerSequence.length() - 1);
          
          // Check if first and last characters match
          if (firstChar == seqFirst && lastChar == seqLast)
          {
            // Count inner character matches
            int innerMatches = countInnerMatches(word, lowerSequence);
            // Score based on inner matches + high base score for matching endpoints
            int score = 1000 + (innerMatches * 100);
            candidates.add(new WordCandidate(word, score * frequency));
          }
          else if (couldBeFormedFrom(word, lowerSequence))
          {
            // Standard swipe scoring for other potential matches
            int score = calculateSwipeScore(word, lowerSequence, frequency);
            if (score > 0)
            {
              candidates.add(new WordCandidate(word, score * frequency));
            }
          }
        }
      }
      else
      {
        // Regular typing - strict length matching
        if (Math.abs(word.length() - lowerSequence.length()) > 2)
          continue;
          
        int score = calculateMatchScore(word, lowerSequence);
        if (score > 0)
        {
          candidates.add(new WordCandidate(word, score * frequency));
        }
      }
    }
    
    // Sort by score (higher is better)
    Collections.sort(candidates, new Comparator<WordCandidate>() {
      @Override
      public int compare(WordCandidate a, WordCandidate b) {
        return Integer.compare(b.score, a.score);
      }
    });
    
    // Return top predictions
    List<String> predictions = new ArrayList<>();
    for (int i = 0; i < Math.min(candidates.size(), MAX_PREDICTIONS); i++)
    {
      predictions.add(candidates.get(i).word);
    }
    
    return predictions;
  }
  
  /**
   * Calculate how well a word matches the key sequence
   */
  private int calculateMatchScore(String word, String keySequence)
  {
    // Direct match is highest score
    if (word.equals(keySequence))
      return 1000;
      
    // Check if word starts with the key sequence (prefix match)
    if (word.startsWith(keySequence))
      return 800;
      
    // Check if key sequence starts with word (over-swipe)
    if (keySequence.startsWith(word))
      return 700;
      
    // Calculate edit distance with adjacency consideration
    int distance = calculateEditDistance(word, keySequence);
    if (distance <= MAX_EDIT_DISTANCE)
    {
      return 500 - (distance * 100);
    }
    
    // Fuzzy match based on common characters
    int commonChars = countCommonCharacters(word, keySequence);
    if (commonChars >= Math.min(word.length(), keySequence.length()) - 1)
    {
      return 200 + commonChars * 10;
    }
    
    return 0;
  }
  
  /**
   * Calculate edit distance between two strings
   */
  private int calculateEditDistance(String s1, String s2)
  {
    int[][] dp = new int[s1.length() + 1][s2.length() + 1];
    
    for (int i = 0; i <= s1.length(); i++)
      dp[i][0] = i;
    for (int j = 0; j <= s2.length(); j++)
      dp[0][j] = j;
      
    for (int i = 1; i <= s1.length(); i++)
    {
      for (int j = 1; j <= s2.length(); j++)
      {
        char c1 = s1.charAt(i - 1);
        char c2 = s2.charAt(j - 1);
        
        if (c1 == c2)
        {
          dp[i][j] = dp[i - 1][j - 1];
        }
        else
        {
          // Check if keys are adjacent (common swipe error)
          boolean adjacent = isAdjacent(c1, c2);
          int substitutionCost = adjacent ? 1 : 2;
          
          dp[i][j] = Math.min(
            dp[i - 1][j] + 1,  // deletion
            Math.min(
              dp[i][j - 1] + 1,  // insertion
              dp[i - 1][j - 1] + substitutionCost  // substitution
            )
          );
        }
      }
    }
    
    return dp[s1.length()][s2.length()];
  }
  
  /**
   * Check if two keys are adjacent on the keyboard
   */
  private boolean isAdjacent(char c1, char c2)
  {
    List<Character> adjacent = _adjacentKeys.get(c1);
    return adjacent != null && adjacent.contains(c2);
  }
  
  /**
   * Count common characters between two strings
   */
  private int countCommonCharacters(String s1, String s2)
  {
    int count = 0;
    int j = 0;
    for (int i = 0; i < s1.length() && j < s2.length(); i++)
    {
      if (s1.charAt(i) == s2.charAt(j))
      {
        count++;
        j++;
      }
      else if (j + 1 < s2.length() && s1.charAt(i) == s2.charAt(j + 1))
      {
        count++;
        j += 2;
      }
    }
    return count;
  }
  
  /**
   * Check if a word could be formed from a swipe sequence
   * All letters of the word should appear in order in the sequence
   */
  private boolean couldBeFormedFrom(String word, String sequence)
  {
    int seqIndex = 0;
    for (char c : word.toCharArray())
    {
      // Find this character in the sequence starting from current position
      int found = sequence.indexOf(c, seqIndex);
      if (found == -1)
        return false;
      seqIndex = found + 1;
    }
    return true;
  }
  
  /**
   * Calculate score for swipe sequence matching
   */
  private int calculateSwipeScore(String word, String sequence, int frequency)
  {
    // Find positions of word characters in sequence
    int seqIndex = 0;
    int totalGaps = 0;
    int matchedChars = 0;
    
    for (char c : word.toCharArray())
    {
      int found = sequence.indexOf(c, seqIndex);
      if (found != -1)
      {
        totalGaps += (found - seqIndex);
        seqIndex = found + 1;
        matchedChars++;
      }
    }
    
    // Score based on:
    // 1. How many characters matched
    // 2. How much of the sequence was used (less gaps = better)
    // 3. Word frequency
    float matchRatio = (float)matchedChars / word.length();
    float gapPenalty = 1.0f / (1.0f + totalGaps / 10.0f);
    
    return (int)(matchRatio * gapPenalty * 1000);
  }
  
  /**
   * Get dictionary size
   */
  public int getDictionarySize()
  {
    return _dictionary.size();
  }
  
  /**
   * Count matching characters in the middle of the word (excluding first/last)
   */
  private int countInnerMatches(String word, String sequence)
  {
    if (word.length() <= 2 || sequence.length() <= 2)
      return 0;
      
    int matches = 0;
    int seqIndex = 1; // Start after first character
    
    // Check inner characters (skip first and last)
    for (int i = 1; i < word.length() - 1; i++)
    {
      char c = word.charAt(i);
      // Look for this character in the remaining sequence
      for (int j = seqIndex; j < sequence.length() - 1; j++)
      {
        if (sequence.charAt(j) == c)
        {
          matches++;
          seqIndex = j + 1;
          break;
        }
      }
    }
    
    return matches;
  }
  
  /**
   * Helper class to store word candidates with scores
   */
  private static class WordCandidate
  {
    final String word;
    final int score;
    
    WordCandidate(String word, int score)
    {
      this.word = word;
      this.score = score;
    }
  }
}