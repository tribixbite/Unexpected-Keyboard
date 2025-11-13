package juloo.keyboard2;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Manages contraction mappings for apostrophe insertion in predictions.
 *
 * Handles two types of contractions:
 * 1. Non-paired: apostrophe-free forms that map to single contractions
 *    Example: "dont" -> "don't", "cant" -> "can't"
 * 2. Paired: base words that have multiple contraction variants
 *    Example: "well" -> ["we'll", "well"], "id" -> ["I'd", "id"]
 *
 * This class is extracted from Keyboard2.java for better separation of concerns
 * and testability (v1.32.341).
 */
public class ContractionManager
{
  private static final String TAG = "ContractionManager";

  // Non-paired contractions: apostrophe-free form -> contraction with apostrophe
  // Example: "dont" -> "don't", "wholl" -> "who'll"
  private final Map<String, String> _nonPairedContractions;

  // Set of all known contractions (both non-paired and paired) for quick lookup
  // Used to identify contractions in predictions and prevent unwanted autocorrect
  private final Set<String> _knownContractions;

  private final AssetManager _assetManager;

  /**
   * Creates a new ContractionManager.
   *
   * @param context Android context for accessing asset files
   */
  public ContractionManager(Context context)
  {
    _nonPairedContractions = new HashMap<>();
    _knownContractions = new HashSet<>();
    _assetManager = context.getAssets();
  }

  /**
   * Loads contraction mappings from JSON files in assets/dictionaries/.
   *
   * This method reads:
   * - contractions_non_paired.json: Apostrophe-free forms -> contractions
   * - contraction_pairings.json: Base words -> contraction variants
   *
   * Must be called before using isKnownContraction() or getNonPairedMapping().
   */
  public void loadMappings()
  {
    try
    {
      loadNonPairedContractions();
      loadPairedContractions();

      Log.d(TAG, String.format("Loaded %d non-paired contractions, %d total known contractions",
        _nonPairedContractions.size(), _knownContractions.size()));
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to load contraction mappings", e);
    }
  }

  /**
   * Checks if a word is a known contraction (has apostrophe).
   *
   * @param word Word to check (case-insensitive)
   * @return true if word is in the known contractions set
   *
   * Examples:
   * - isKnownContraction("don't") -> true
   * - isKnownContraction("we'll") -> true
   * - isKnownContraction("hello") -> false
   */
  public boolean isKnownContraction(String word)
  {
    return _knownContractions.contains(word.toLowerCase());
  }

  /**
   * Gets the contraction with apostrophe for an apostrophe-free form.
   *
   * Only works for non-paired contractions (where the apostrophe-free form
   * is not a valid English word).
   *
   * @param withoutApostrophe Apostrophe-free form (case-insensitive)
   * @return Contraction with apostrophe, or null if not found
   *
   * Examples:
   * - getNonPairedMapping("dont") -> "don't"
   * - getNonPairedMapping("wholl") -> "who'll"
   * - getNonPairedMapping("well") -> null (paired contraction)
   */
  public String getNonPairedMapping(String withoutApostrophe)
  {
    return _nonPairedContractions.get(withoutApostrophe.toLowerCase());
  }

  /**
   * Gets the number of non-paired contractions loaded.
   * Useful for testing and diagnostics.
   */
  public int getNonPairedCount()
  {
    return _nonPairedContractions.size();
  }

  /**
   * Gets the total number of known contractions (non-paired + paired).
   * Useful for testing and diagnostics.
   */
  public int getTotalKnownCount()
  {
    return _knownContractions.size();
  }

  /**
   * Loads non-paired contractions from contractions_non_paired.json.
   *
   * Format: {"dont": "don't", "cant": "can't", ...}
   *
   * These are apostrophe-free forms that are NOT valid English words.
   * The neural network predicts "dont", we replace with "don't".
   */
  private void loadNonPairedContractions() throws Exception
  {
    InputStream inputStream = _assetManager.open("dictionaries/contractions_non_paired.json");
    String jsonString = readStream(inputStream);

    JSONObject jsonObj = new JSONObject(jsonString);
    Iterator<String> keys = jsonObj.keys();

    while (keys.hasNext())
    {
      String withoutApostrophe = keys.next();
      String withApostrophe = jsonObj.getString(withoutApostrophe);

      _nonPairedContractions.put(withoutApostrophe.toLowerCase(), withApostrophe.toLowerCase());
      _knownContractions.add(withApostrophe.toLowerCase());
    }

    Log.d(TAG, "Loaded " + _nonPairedContractions.size() + " non-paired contractions");
  }

  /**
   * Loads paired contractions from contraction_pairings.json.
   *
   * Format: {"well": [{"contraction": "we'll", "frequency": 243}], ...}
   *
   * These are base words that ARE valid English words but also have
   * contraction variants. Both forms should appear in predictions.
   * Example: "well" (adverb) and "we'll" (we will) are both valid.
   */
  private void loadPairedContractions() throws Exception
  {
    InputStream inputStream = _assetManager.open("dictionaries/contraction_pairings.json");
    String jsonString = readStream(inputStream);

    JSONObject jsonObj = new JSONObject(jsonString);
    Iterator<String> keys = jsonObj.keys();
    int pairedCount = 0;

    while (keys.hasNext())
    {
      String baseWord = keys.next();
      JSONArray contractions = jsonObj.getJSONArray(baseWord);

      for (int i = 0; i < contractions.length(); i++)
      {
        JSONObject contractionObj = contractions.getJSONObject(i);
        String contraction = contractionObj.getString("contraction");

        _knownContractions.add(contraction.toLowerCase());
        pairedCount++;
      }
    }

    Log.d(TAG, "Loaded " + pairedCount + " paired contractions");
  }

  /**
   * Reads an InputStream into a String.
   * Helper method for JSON file loading.
   */
  private String readStream(InputStream inputStream) throws Exception
  {
    BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
    StringBuilder builder = new StringBuilder();
    String line;

    while ((line = reader.readLine()) != null)
    {
      builder.append(line);
    }

    reader.close();
    return builder.toString();
  }
}
