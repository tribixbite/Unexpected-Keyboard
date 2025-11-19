package juloo.keyboard2;

import android.content.Context;
import android.util.Log;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

/**
 * Tokenizer for neural swipe prediction
 * Handles character-to-index mapping for ONNX model input
 * Matches the tokenizer configuration from the web demo
 */
public class SwipeTokenizer
{
  private static final String TAG = "SwipeTokenizer";
  
  // Special token indices (matching web demo)
  public static final int PAD_IDX = 0;
  public static final int UNK_IDX = 1;
  public static final int SOS_IDX = 2;
  public static final int EOS_IDX = 3;
  
  // Character mappings
  private Map<Character, Integer> _charToIdx;
  private Map<Integer, Character> _idxToChar;
  private boolean _isLoaded = false;
  
  public SwipeTokenizer()
  {
    // Mappings are now loaded from JSON
  }
  
  // Helper class for Gson parsing
  private static class TokenizerConfig {
      Map<String, Integer> char_to_idx;
      Map<String, String> idx_to_char;
  }

  /**
   * Load tokenizer configuration from assets
   */
  public boolean loadFromAssets(Context context)
  {
    try
    {
      Log.d(TAG, "Loading tokenizer configuration from assets");
      
      InputStream inputStream = context.getAssets().open("models/tokenizer_config.json");
      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
      
      Gson gson = new Gson();
      TokenizerConfig config = gson.fromJson(reader, TokenizerConfig.class);
      reader.close();
      
      _charToIdx = new HashMap<>();
      for (Map.Entry<String, Integer> entry : config.char_to_idx.entrySet()) {
          if (entry.getKey().length() > 0) {
              _charToIdx.put(entry.getKey().charAt(0), entry.getValue());
          }
      }

      _idxToChar = new HashMap<>();
      for (Map.Entry<String, String> entry : config.idx_to_char.entrySet()) {
          if (entry.getValue().length() > 0) {
              _idxToChar.put(Integer.parseInt(entry.getKey()), entry.getValue().charAt(0));
          }
      }
      
      _isLoaded = true;
      Log.d(TAG, String.format("Tokenizer loaded with %d characters", _charToIdx.size()));
      return true;
    }
    catch (IOException e)
    {
      Log.w(TAG, "Could not load tokenizer from assets, using defaults: " + e.getMessage());
      _isLoaded = false;
      return false;
    }
  }
  
  /**
   * Convert character to token index
   */
  public int charToIndex(char c)
  {
    Character ch = Character.toLowerCase(c);
    Integer idx = _charToIdx.get(ch);
    return idx != null ? idx : UNK_IDX;
  }
  
  /**
   * Convert token index to character
   */
  public char indexToChar(int idx)
  {
    Character ch = _idxToChar.get(idx);
    return ch != null ? ch : '?';
  }
  
  /**
   * Get vocabulary size
   */
  public int getVocabSize()
  {
    return _charToIdx.size();
  }
  
  /**
   * Check if tokenizer is loaded
   */
  public boolean isLoaded()
  {
    return _isLoaded;
  }
  
  private void addMapping(int idx, char ch)
  {
    _charToIdx.put(ch, idx);
    _idxToChar.put(idx, ch);
  }
  
  /**
   * Get character-to-index mapping (for debugging)
   */
  public Map<Character, Integer> getCharToIdxMapping()
  {
    return new HashMap<>(_charToIdx);
  }
}