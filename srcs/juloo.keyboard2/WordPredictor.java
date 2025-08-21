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
    
    // Try enhanced dictionary first
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
      // Try fallback dictionary
      try
      {
        BufferedReader reader = new BufferedReader(
          new InputStreamReader(context.getAssets().open(fallbackFilename)));
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
        android.util.Log.d("WordPredictor", "Loaded basic dictionary: " + fallbackFilename + " with " + _dictionary.size() + " words");
      }
      catch (IOException e2)
      {
        // If neither dictionary file exists, use a basic built-in dictionary
        loadBasicDictionary();
        android.util.Log.d("WordPredictor", "Using built-in dictionary with " + _dictionary.size() + " words");
      }
    }
  }
  
  /**
   * Load a basic built-in dictionary for testing
   */
  private void loadBasicDictionary()
  {
    // Common English words with frequencies
    _dictionary.put("the", 10000);
    _dictionary.put("and", 9000);
    _dictionary.put("you", 8500);
    _dictionary.put("that", 8000);
    _dictionary.put("was", 7500);
    _dictionary.put("for", 7000);
    _dictionary.put("are", 6500);
    _dictionary.put("with", 6000);
    _dictionary.put("his", 5500);
    _dictionary.put("they", 5000);
    _dictionary.put("this", 4500);
    _dictionary.put("have", 4000);
    _dictionary.put("from", 3500);
    _dictionary.put("word", 3000);
    _dictionary.put("but", 2500);
    _dictionary.put("what", 2000);
    _dictionary.put("some", 1500);
    _dictionary.put("can", 1000);
    _dictionary.put("hello", 900);
    _dictionary.put("world", 800);
    _dictionary.put("test", 700);
    _dictionary.put("type", 600);
    _dictionary.put("keyboard", 500);
    _dictionary.put("swipe", 400);
    
    // Add more common words for better testing
    _dictionary.put("is", 9500);
    _dictionary.put("it", 9000);
    _dictionary.put("be", 8500);
    _dictionary.put("of", 8000);
    _dictionary.put("to", 7500);
    _dictionary.put("in", 7000);
    _dictionary.put("he", 6500);
    _dictionary.put("has", 6000);
    _dictionary.put("will", 5500);
    _dictionary.put("on", 5000);
    _dictionary.put("do", 4500);
    _dictionary.put("say", 4000);
    _dictionary.put("she", 3500);
    _dictionary.put("an", 3000);
    _dictionary.put("or", 2500);
    _dictionary.put("by", 2000);
    _dictionary.put("as", 1500);
    _dictionary.put("we", 1000);
    _dictionary.put("her", 900);
    _dictionary.put("out", 800);
    _dictionary.put("if", 700);
    _dictionary.put("would", 600);
    _dictionary.put("there", 500);
    _dictionary.put("their", 400);
    _dictionary.put("been", 300);
    _dictionary.put("when", 200);
    _dictionary.put("who", 100);
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
      
      // For swipe sequences, be more lenient with length differences
      if (!isSwipeSequence && Math.abs(word.length() - lowerSequence.length()) > 2)
        continue;
      
      // For swipe sequences, check if word could be formed from the sequence
      if (isSwipeSequence && !couldBeFormedFrom(word, lowerSequence))
        continue;
        
      // Calculate match score
      int score = isSwipeSequence ? 
        calculateSwipeScore(word, lowerSequence, frequency) :
        calculateMatchScore(word, lowerSequence);
      
      if (score > 0)
      {
        candidates.add(new WordCandidate(word, score * frequency));
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