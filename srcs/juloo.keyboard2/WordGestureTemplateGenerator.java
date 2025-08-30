package juloo.keyboard2;

import android.content.Context;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates gesture templates for words based on QWERTY keyboard layout
 * Converts words from dictionary into gesture patterns for CGR recognition
 */
public class WordGestureTemplateGenerator
{
  // QWERTY keyboard layout coordinates (normalized to 1000x1000 space)
  private static final Map<Character, ContinuousGestureRecognizer.Point> QWERTY_COORDS;
  
  static
  {
    QWERTY_COORDS = new HashMap<>();
    
    // Top row (QWERTYUIOP) - Fixed positive coordinates in 1000x1000 space
    double topY = 200;
    double[] topX = {100, 200, 300, 400, 500, 600, 700, 800, 900, 950};
    char[] topChars = {'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'};
    for (int i = 0; i < topChars.length; i++)
    {
      QWERTY_COORDS.put(topChars[i], new ContinuousGestureRecognizer.Point(topX[i], topY));
    }
    
    // Middle row (ASDFGHJKL)
    double middleY = 500;
    double[] middleX = {150, 250, 350, 450, 550, 650, 750, 850, 950};
    char[] middleChars = {'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'};
    for (int i = 0; i < middleChars.length; i++)
    {
      QWERTY_COORDS.put(middleChars[i], new ContinuousGestureRecognizer.Point(middleX[i], middleY));
    }
    
    // Bottom row (ZXCVBNM)
    double bottomY = 800;
    double[] bottomX = {200, 300, 400, 500, 600, 700, 800};
    char[] bottomChars = {'z', 'x', 'c', 'v', 'b', 'n', 'm'};
    for (int i = 0; i < bottomChars.length; i++)
    {
      QWERTY_COORDS.put(bottomChars[i], new ContinuousGestureRecognizer.Point(bottomX[i], bottomY));
    }
  }
  
  private final List<String> dictionary;
  private final Map<String, Integer> wordFrequencies;
  
  public WordGestureTemplateGenerator()
  {
    dictionary = new ArrayList<>();
    wordFrequencies = new HashMap<>();
  }
  
  /**
   * Load dictionary from en.txt file
   */
  public void loadDictionary(Context context)
  {
    dictionary.clear();
    wordFrequencies.clear();
    
    try
    {
      BufferedReader reader = new BufferedReader(
        new InputStreamReader(context.getAssets().open("dictionaries/en.txt")));
      String line;
      int wordCount = 0;
      
      while ((line = reader.readLine()) != null && wordCount < 5000) // Limit to top 5000 words
      {
        // Skip comments and empty lines
        if (line.startsWith("#") || line.trim().isEmpty()) continue;
        
        String word = line.trim().toLowerCase();
        
        // Only include words with letters (3-12 characters for comprehensive gesture typing)
        if (word.matches("[a-z]+") && word.length() >= 3 && word.length() <= 12)
        {
          dictionary.add(word);
          wordFrequencies.put(word, 1000); // Default frequency since no frequencies in new format
          wordCount++;
        }
      }
      reader.close();
      
      android.util.Log.d("WordGestureTemplateGenerator", 
        "Loaded " + dictionary.size() + " words for gesture templates");
    }
    catch (IOException e)
    {
      android.util.Log.e("WordGestureTemplateGenerator", "Failed to load dictionary: " + e.getMessage());
    }
  }
  
  /**
   * Generate gesture template for a single word
   */
  public ContinuousGestureRecognizer.Template generateWordTemplate(String word)
  {
    word = word.toLowerCase();
    List<ContinuousGestureRecognizer.Point> points = new ArrayList<>();
    
    for (char c : word.toCharArray())
    {
      ContinuousGestureRecognizer.Point coord = QWERTY_COORDS.get(c);
      if (coord != null)
      {
        points.add(new ContinuousGestureRecognizer.Point(coord.x, coord.y));
      }
      else
      {
        android.util.Log.w("WordGestureTemplateGenerator", 
          "No coordinate found for character: " + c);
        return null; // Skip words with unknown characters
      }
    }
    
    if (points.size() < 2)
    {
      return null; // Need at least 2 points for a gesture
    }
    
    return new ContinuousGestureRecognizer.Template(word, points);
  }
  
  /**
   * Generate templates for all dictionary words
   */
  public List<ContinuousGestureRecognizer.Template> generateAllWordTemplates()
  {
    List<ContinuousGestureRecognizer.Template> templates = new ArrayList<>();
    int successCount = 0;
    
    for (String word : dictionary)
    {
      ContinuousGestureRecognizer.Template template = generateWordTemplate(word);
      if (template != null)
      {
        templates.add(template);
        successCount++;
      }
    }
    
    android.util.Log.d("WordGestureTemplateGenerator", 
      "Generated " + successCount + " word templates from " + dictionary.size() + " dictionary words");
    
    return templates;
  }
  
  /**
   * Generate templates for most frequent words only
   */
  public List<ContinuousGestureRecognizer.Template> generateFrequentWordTemplates(int maxWords)
  {
    List<ContinuousGestureRecognizer.Template> templates = new ArrayList<>();
    int count = 0;
    
    for (String word : dictionary)
    {
      if (count >= maxWords) break;
      
      ContinuousGestureRecognizer.Template template = generateWordTemplate(word);
      if (template != null)
      {
        templates.add(template);
        count++;
      }
    }
    
    android.util.Log.d("WordGestureTemplateGenerator", 
      "Generated " + count + " frequent word templates");
    
    return templates;
  }
  
  /**
   * Get word frequency for a given word
   */
  public int getWordFrequency(String word)
  {
    return wordFrequencies.getOrDefault(word.toLowerCase(), 0);
  }
  
  /**
   * Check if word is in dictionary
   */
  public boolean isWordInDictionary(String word)
  {
    return wordFrequencies.containsKey(word.toLowerCase());
  }
  
  /**
   * Get dictionary size
   */
  public int getDictionarySize()
  {
    return dictionary.size();
  }
  
  /**
   * Get coordinate for a character
   */
  public static ContinuousGestureRecognizer.Point getCharacterCoordinate(char c)
  {
    ContinuousGestureRecognizer.Point coord = QWERTY_COORDS.get(Character.toLowerCase(c));
    return coord != null ? new ContinuousGestureRecognizer.Point(coord.x, coord.y) : null;
  }
  
  /**
   * Calculate gesture path length for a word (for complexity estimation)
   */
  public double calculateGesturePathLength(String word)
  {
    word = word.toLowerCase();
    double totalLength = 0.0;
    ContinuousGestureRecognizer.Point prevPoint = null;
    
    for (char c : word.toCharArray())
    {
      ContinuousGestureRecognizer.Point point = QWERTY_COORDS.get(c);
      if (point != null)
      {
        if (prevPoint != null)
        {
          double dx = point.x - prevPoint.x;
          double dy = point.y - prevPoint.y;
          totalLength += Math.sqrt(dx * dx + dy * dy);
        }
        prevPoint = point;
      }
    }
    
    return totalLength;
  }
  
  /**
   * Get words by length range
   */
  public List<String> getWordsByLength(int minLength, int maxLength)
  {
    List<String> filtered = new ArrayList<>();
    
    for (String word : dictionary)
    {
      if (word.length() >= minLength && word.length() <= maxLength)
      {
        filtered.add(word);
      }
    }
    
    return filtered;
  }
  
  /**
   * Generate templates with complexity filtering
   * Excludes words that would create overly complex or simple gestures
   */
  public List<ContinuousGestureRecognizer.Template> generateBalancedWordTemplates(int maxWords)
  {
    List<ContinuousGestureRecognizer.Template> templates = new ArrayList<>();
    int count = 0;
    
    for (String word : dictionary)
    {
      if (count >= maxWords) break;
      
      // Filter by gesture complexity
      double pathLength = calculateGesturePathLength(word);
      if (pathLength > 100 && pathLength < 3000) // Reasonable gesture complexity
      {
        ContinuousGestureRecognizer.Template template = generateWordTemplate(word);
        if (template != null)
        {
          templates.add(template);
          count++;
        }
      }
    }
    
    android.util.Log.d("WordGestureTemplateGenerator", 
      "Generated " + count + " balanced word templates");
    
    return templates;
  }
}