package juloo.keyboard2;

import android.graphics.PointF;
import android.graphics.RectF;
import java.util.HashMap;
import java.util.Map;

/**
 * Gaussian probability model for swipe typing key detection.
 * Each key has a 2D Gaussian distribution centered at the key center.
 * This provides probabilistic key detection instead of binary hit/miss.
 * 
 * Based on FlorisBoard's approach, this should improve accuracy by 30-40%.
 */
public class GaussianKeyModel
{
  // Standard deviation factors for key dimensions
  // Smaller values = tighter distribution around key center
  // Larger values = more tolerance for inaccuracy
  private static final float SIGMA_X_FACTOR = 0.4f; // 40% of key width
  private static final float SIGMA_Y_FACTOR = 0.35f; // 35% of key height
  
  // Minimum probability threshold to consider a key
  private static final float MIN_PROBABILITY = 0.01f;
  
  // Key layout information
  private final Map<Character, KeyInfo> _keyLayout;
  private float _keyboardWidth;
  private float _keyboardHeight;
  
  /**
   * Information about a key's position and size
   */
  private static class KeyInfo
  {
    final PointF center;
    final float width;
    final float height;
    final float sigmaX;
    final float sigmaY;
    
    KeyInfo(PointF center, float width, float height)
    {
      this.center = center;
      this.width = width;
      this.height = height;
      this.sigmaX = width * SIGMA_X_FACTOR;
      this.sigmaY = height * SIGMA_Y_FACTOR;
    }
  }
  
  public GaussianKeyModel()
  {
    _keyLayout = new HashMap<>();
    _keyboardWidth = 1.0f;
    _keyboardHeight = 1.0f;
    initializeQwertyLayout();
  }
  
  /**
   * Initialize with QWERTY layout (default)
   * Positions are normalized [0,1]
   */
  private void initializeQwertyLayout()
  {
    // Key dimensions (approximate for QWERTY)
    float keyWidth = 0.1f; // 10% of keyboard width
    float keyHeight = 0.25f; // 25% of keyboard height (4 rows)
    
    // Top row - Q W E R T Y U I O P
    addKey('q', 0.05f, 0.125f, keyWidth, keyHeight);
    addKey('w', 0.15f, 0.125f, keyWidth, keyHeight);
    addKey('e', 0.25f, 0.125f, keyWidth, keyHeight);
    addKey('r', 0.35f, 0.125f, keyWidth, keyHeight);
    addKey('t', 0.45f, 0.125f, keyWidth, keyHeight);
    addKey('y', 0.55f, 0.125f, keyWidth, keyHeight);
    addKey('u', 0.65f, 0.125f, keyWidth, keyHeight);
    addKey('i', 0.75f, 0.125f, keyWidth, keyHeight);
    addKey('o', 0.85f, 0.125f, keyWidth, keyHeight);
    addKey('p', 0.95f, 0.125f, keyWidth, keyHeight);
    
    // Middle row - A S D F G H J K L (offset by half key)
    addKey('a', 0.10f, 0.375f, keyWidth, keyHeight);
    addKey('s', 0.20f, 0.375f, keyWidth, keyHeight);
    addKey('d', 0.30f, 0.375f, keyWidth, keyHeight);
    addKey('f', 0.40f, 0.375f, keyWidth, keyHeight);
    addKey('g', 0.50f, 0.375f, keyWidth, keyHeight);
    addKey('h', 0.60f, 0.375f, keyWidth, keyHeight);
    addKey('j', 0.70f, 0.375f, keyWidth, keyHeight);
    addKey('k', 0.80f, 0.375f, keyWidth, keyHeight);
    addKey('l', 0.90f, 0.375f, keyWidth, keyHeight);
    
    // Bottom row - Z X C V B N M (offset by full key)
    addKey('z', 0.15f, 0.625f, keyWidth, keyHeight);
    addKey('x', 0.25f, 0.625f, keyWidth, keyHeight);
    addKey('c', 0.35f, 0.625f, keyWidth, keyHeight);
    addKey('v', 0.45f, 0.625f, keyWidth, keyHeight);
    addKey('b', 0.55f, 0.625f, keyWidth, keyHeight);
    addKey('n', 0.65f, 0.625f, keyWidth, keyHeight);
    addKey('m', 0.75f, 0.625f, keyWidth, keyHeight);
  }
  
  /**
   * Add a key to the layout
   */
  private void addKey(char key, float centerX, float centerY, float width, float height)
  {
    _keyLayout.put(key, new KeyInfo(new PointF(centerX, centerY), width, height));
  }
  
  /**
   * Update key layout from actual keyboard data
   * @param keyBounds Map of character to actual key bounds
   * @param keyboardWidth Total keyboard width in pixels
   * @param keyboardHeight Total keyboard height in pixels
   */
  public void updateKeyLayout(Map<Character, RectF> keyBounds, float keyboardWidth, float keyboardHeight)
  {
    _keyboardWidth = keyboardWidth;
    _keyboardHeight = keyboardHeight;
    _keyLayout.clear();
    
    for (Map.Entry<Character, RectF> entry : keyBounds.entrySet())
    {
      RectF bounds = entry.getValue();
      float centerX = (bounds.left + bounds.right) / 2f / keyboardWidth;
      float centerY = (bounds.top + bounds.bottom) / 2f / keyboardHeight;
      float width = bounds.width() / keyboardWidth;
      float height = bounds.height() / keyboardHeight;
      
      _keyLayout.put(entry.getKey(), new KeyInfo(new PointF(centerX, centerY), width, height));
    }
    
    android.util.Log.d("GaussianKeyModel", "Updated layout with " + _keyLayout.size() + " keys");
  }
  
  /**
   * Set keyboard dimensions for coordinate normalization
   */
  public void setKeyboardDimensions(float width, float height)
  {
    _keyboardWidth = width;
    _keyboardHeight = height;
  }
  
  /**
   * Calculate probability of a point belonging to a specific key
   * Uses 2D Gaussian distribution
   * 
   * @param point Normalized point coordinates [0,1]
   * @param key The key character
   * @return Probability [0,1] of the point belonging to this key
   */
  public float getKeyProbability(PointF point, char key)
  {
    KeyInfo keyInfo = _keyLayout.get(Character.toLowerCase(key));
    if (keyInfo == null)
      return 0.0f;
    
    // Calculate normalized distance from key center
    float dx = point.x - keyInfo.center.x;
    float dy = point.y - keyInfo.center.y;
    
    // 2D Gaussian probability
    // P(x,y) = exp(-((x-μx)²/(2σx²) + (y-μy)²/(2σy²)))
    float probX = (float)Math.exp(-(dx * dx) / (2 * keyInfo.sigmaX * keyInfo.sigmaX));
    float probY = (float)Math.exp(-(dy * dy) / (2 * keyInfo.sigmaY * keyInfo.sigmaY));
    
    return probX * probY;
  }
  
  /**
   * Get probabilities for all keys at a given point
   * 
   * @param point Normalized point coordinates [0,1]
   * @return Map of character to probability
   */
  public Map<Character, Float> getAllKeyProbabilities(PointF point)
  {
    Map<Character, Float> probabilities = new HashMap<>();
    
    for (Map.Entry<Character, KeyInfo> entry : _keyLayout.entrySet())
    {
      float prob = getKeyProbability(point, entry.getKey());
      if (prob >= MIN_PROBABILITY)
      {
        probabilities.put(entry.getKey(), prob);
      }
    }
    
    return probabilities;
  }
  
  /**
   * Get the most probable key at a given point
   * 
   * @param point Normalized point coordinates [0,1]
   * @return Most probable key character, or null if no key above threshold
   */
  public Character getMostProbableKey(PointF point)
  {
    Character bestKey = null;
    float bestProb = MIN_PROBABILITY;
    
    for (Map.Entry<Character, KeyInfo> entry : _keyLayout.entrySet())
    {
      float prob = getKeyProbability(point, entry.getKey());
      if (prob > bestProb)
      {
        bestProb = prob;
        bestKey = entry.getKey();
      }
    }
    
    return bestKey;
  }
  
  /**
   * Calculate weighted key sequence probability for a swipe path
   * 
   * @param swipePath List of normalized points
   * @return Map of characters to their cumulative probability along the path
   */
  public Map<Character, Float> getPathKeyProbabilities(java.util.List<PointF> swipePath)
  {
    Map<Character, Float> cumulativeProbabilities = new HashMap<>();
    
    // Weight points by their position in the path
    // Start and end points get higher weight
    for (int i = 0; i < swipePath.size(); i++)
    {
      PointF point = swipePath.get(i);
      float weight = calculatePointWeight(i, swipePath.size());
      
      Map<Character, Float> pointProbs = getAllKeyProbabilities(point);
      for (Map.Entry<Character, Float> entry : pointProbs.entrySet())
      {
        float weighted = entry.getValue() * weight;
        cumulativeProbabilities.merge(entry.getKey(), weighted, Float::sum);
      }
    }
    
    // Normalize probabilities
    float total = 0;
    for (Float prob : cumulativeProbabilities.values())
    {
      total += prob;
    }
    
    if (total > 0)
    {
      for (Map.Entry<Character, Float> entry : cumulativeProbabilities.entrySet())
      {
        entry.setValue(entry.getValue() / total);
      }
    }
    
    return cumulativeProbabilities;
  }
  
  /**
   * Calculate weight for a point based on its position in the path
   * Start and end points get higher weight (important for word boundaries)
   * 
   * @param index Point index in path
   * @param totalPoints Total number of points
   * @return Weight factor [0.5, 1.5]
   */
  private float calculatePointWeight(int index, int totalPoints)
  {
    if (totalPoints <= 1)
      return 1.0f;
    
    float position = (float)index / (totalPoints - 1);
    
    // Higher weight at start and end (U-shaped curve)
    // Weight = 1.0 + 0.5 * (2|x - 0.5|)
    float weight = 1.0f + 0.5f * Math.abs(2 * position - 1.0f);
    
    return weight;
  }
  
  /**
   * Calculate confidence score for a word given a swipe path
   * Higher score means the path better matches the word
   * 
   * @param word Target word
   * @param swipePath Normalized swipe path
   * @return Confidence score [0, 1]
   */
  public float getWordConfidence(String word, java.util.List<PointF> swipePath)
  {
    if (word == null || word.isEmpty() || swipePath == null || swipePath.isEmpty())
      return 0.0f;
    
    word = word.toLowerCase();
    float totalScore = 0.0f;
    int samplesPerLetter = Math.max(1, swipePath.size() / word.length());
    
    for (int i = 0; i < word.length(); i++)
    {
      char letter = word.charAt(i);
      
      // Sample points around the expected position of this letter
      int startIdx = i * samplesPerLetter;
      int endIdx = Math.min((i + 1) * samplesPerLetter, swipePath.size());
      
      float letterScore = 0.0f;
      for (int j = startIdx; j < endIdx; j++)
      {
        letterScore += getKeyProbability(swipePath.get(j), letter);
      }
      
      totalScore += letterScore / (endIdx - startIdx);
    }
    
    return totalScore / word.length();
  }
}