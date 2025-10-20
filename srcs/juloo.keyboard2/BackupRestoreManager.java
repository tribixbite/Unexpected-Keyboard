package juloo.keyboard2;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;
import android.util.DisplayMetrics;
import android.util.Log;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Manages backup and restore of keyboard configuration
 * Uses Storage Access Framework (SAF) for Android 15+ compatibility
 */
public class BackupRestoreManager
{
  private static final String TAG = "BackupRestoreManager";
  private final Context context;
  private final Gson gson;

  public BackupRestoreManager(Context context)
  {
    this.context = context;
    this.gson = new GsonBuilder().setPrettyPrinting().create();
  }

  /**
   * Export all preferences to JSON file
   * @param uri URI from Storage Access Framework (ACTION_CREATE_DOCUMENT)
   * @return true if successful
   */
  public boolean exportConfig(Uri uri, SharedPreferences prefs) throws Exception
  {
    try
    {
      // Collect metadata
      JsonObject root = new JsonObject();
      JsonObject metadata = new JsonObject();

      // App version
      String versionName = context.getPackageManager()
        .getPackageInfo(context.getPackageName(), 0).versionName;
      int versionCode = context.getPackageManager()
        .getPackageInfo(context.getPackageName(), 0).versionCode;

      metadata.addProperty("app_version", versionName);
      metadata.addProperty("version_code", versionCode);
      metadata.addProperty("export_date",
        new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(new Date()));

      // Screen dimensions
      DisplayMetrics dm = context.getResources().getDisplayMetrics();
      metadata.addProperty("screen_width", dm.widthPixels);
      metadata.addProperty("screen_height", dm.heightPixels);
      metadata.addProperty("screen_density", dm.density);
      metadata.addProperty("android_version", android.os.Build.VERSION.SDK_INT);

      root.add("metadata", metadata);

      // Export all preferences
      Map<String, ?> allPrefs = prefs.getAll();
      JsonObject preferences = new JsonObject();

      for (Map.Entry<String, ?> entry : allPrefs.entrySet())
      {
        String key = entry.getKey();
        Object value = entry.getValue();

        // Preserve JSON-string preferences (layouts, extra_keys, custom_extra_keys)
        // These are already stored as JSON strings and should be preserved as-is
        if (isJsonStringPreference(key) && value instanceof String)
        {
          try
          {
            // Parse the JSON string and add as JsonElement to avoid double-encoding
            preferences.add(key, JsonParser.parseString((String)value));
          }
          catch (Exception e)
          {
            Log.w(TAG, "Failed to parse JSON preference: " + key, e);
            // Fall back to regular serialization if parsing fails
            preferences.add(key, gson.toJsonTree(value));
          }
        }
        else if (isInternalPreference(key))
        {
          // Skip internal state preferences
          Log.i(TAG, "Skipping internal preference on export: " + key);
        }
        else
        {
          preferences.add(key, gson.toJsonTree(value));
        }
      }

      root.add("preferences", preferences);

      // Write to file
      try (OutputStream outputStream = context.getContentResolver().openOutputStream(uri);
           OutputStreamWriter writer = new OutputStreamWriter(outputStream))
      {
        gson.toJson(root, writer);
        writer.flush();
      }

      Log.i(TAG, "Exported " + allPrefs.size() + " preferences");
      return true;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Export failed", e);
      throw new Exception("Export failed: " + e.getMessage(), e);
    }
  }

  /**
   * Import preferences from JSON file with version-tolerant parsing
   * @param uri URI from Storage Access Framework (ACTION_OPEN_DOCUMENT)
   * @return ImportResult with statistics
   */
  public ImportResult importConfig(Uri uri, SharedPreferences prefs) throws Exception
  {
    try
    {
      // Read JSON file
      StringBuilder jsonBuilder = new StringBuilder();
      try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(context.getContentResolver().openInputStream(uri))))
      {
        String line;
        while ((line = reader.readLine()) != null)
        {
          jsonBuilder.append(line);
        }
      }

      JsonObject root = JsonParser.parseString(jsonBuilder.toString()).getAsJsonObject();

      // Parse metadata (optional, for informational purposes)
      ImportResult result = new ImportResult();
      if (root.has("metadata"))
      {
        JsonObject metadata = root.getAsJsonObject("metadata");
        result.sourceVersion = metadata.has("app_version") ?
          metadata.get("app_version").getAsString() : "unknown";
        result.sourceScreenWidth = metadata.has("screen_width") ?
          metadata.get("screen_width").getAsInt() : 0;
        result.sourceScreenHeight = metadata.has("screen_height") ?
          metadata.get("screen_height").getAsInt() : 0;
      }

      // Get current screen dimensions for comparison
      DisplayMetrics dm = context.getResources().getDisplayMetrics();
      result.currentScreenWidth = dm.widthPixels;
      result.currentScreenHeight = dm.heightPixels;

      // Import preferences with validation
      if (!root.has("preferences"))
      {
        throw new Exception("Invalid backup file: missing preferences section");
      }

      JsonObject preferences = root.getAsJsonObject("preferences");
      SharedPreferences.Editor editor = prefs.edit();

      int imported = 0;
      int skipped = 0;

      for (Map.Entry<String, com.google.gson.JsonElement> entry : preferences.entrySet())
      {
        String key = entry.getKey();
        com.google.gson.JsonElement value = entry.getValue();

        try
        {
          if (importPreference(editor, key, value))
          {
            imported++;
            result.importedKeys.add(key);
          }
          else
          {
            skipped++;
            result.skippedKeys.add(key);
          }
        }
        catch (Exception e)
        {
          Log.w(TAG, "Failed to import key: " + key, e);
          skipped++;
          result.skippedKeys.add(key);
        }
      }

      editor.apply();

      result.importedCount = imported;
      result.skippedCount = skipped;

      Log.i(TAG, "Import complete: " + imported + " imported, " + skipped + " skipped");
      return result;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Import failed", e);
      throw new Exception("Import failed: " + e.getMessage(), e);
    }
  }

  /**
   * Import a single preference with type detection and validation
   * @return true if imported, false if skipped
   */
  private boolean importPreference(SharedPreferences.Editor editor, String key,
                                    com.google.gson.JsonElement value)
  {
    // Skip internal state preferences
    if (isInternalPreference(key))
    {
      Log.i(TAG, "Skipping internal preference: " + key);
      return false;
    }

    // Handle JSON-string preferences (layouts, extra_keys, custom_extra_keys)
    // These are stored as JSON strings in SharedPreferences
    if (isJsonStringPreference(key))
    {
      String jsonString;

      // Handle both old double-encoded strings and new native JSON arrays/objects
      if (value.isJsonPrimitive() && value.getAsJsonPrimitive().isString())
      {
        // Old format: the JSON was exported as a string primitive (double-encoded)
        // Just use the string value directly
        jsonString = value.getAsString();
        Log.i(TAG, "Importing old-format JSON-string preference: " + key);
      }
      else if (value.isJsonArray() || value.isJsonObject())
      {
        // New format: the JSON is a native array/object
        // Convert to compact JSON string (no pretty printing for preferences)
        jsonString = value.toString();
        Log.i(TAG, "Importing new-format JSON-string preference: " + key);
      }
      else
      {
        Log.w(TAG, "Unexpected format for JSON-string preference: " + key);
        return false;
      }

      editor.putString(key, jsonString);
      return true;
    }
    // Handle different preference types
    if (value.isJsonPrimitive())
    {
      com.google.gson.JsonPrimitive primitive = value.getAsJsonPrimitive();

      if (primitive.isBoolean())
      {
        editor.putBoolean(key, primitive.getAsBoolean());
        return true;
      }
      else if (primitive.isNumber())
      {
        // Check if this preference is known to be a float type
        // We must use the correct type because SharedPreferences throws ClassCastException
        // if we try to read an int as float or vice versa
        if (isFloatPreference(key))
        {
          // Known float preference
          float floatValue = primitive.getAsFloat();
          if (validateFloatPreference(key, floatValue))
          {
            editor.putFloat(key, floatValue);
            return true;
          }
          else
          {
            Log.w(TAG, "Skipping invalid float value for " + key + ": " + floatValue);
            return false;
          }
        }
        else
        {
          // Assume integer for all other numeric preferences
          int intValue = primitive.getAsInt();
          if (validateIntPreference(key, intValue))
          {
            editor.putInt(key, intValue);
            return true;
          }
          else
          {
            Log.w(TAG, "Skipping invalid int value for " + key + ": " + intValue);
            return false;
          }
        }
      }
      else if (primitive.isString())
      {
        String stringValue = primitive.getAsString();

        // Some preferences store integers as strings (from ListPreference)
        // Parse and store them as actual integers to prevent ClassCastException
        if (isIntegerStoredAsString(key))
        {
          try
          {
            int intValue = Integer.parseInt(stringValue);
            if (validateIntPreference(key, intValue))
            {
              editor.putInt(key, intValue);
              return true;
            }
            else
            {
              Log.w(TAG, "Skipping invalid int-as-string value for " + key + ": " + intValue);
              return false;
            }
          }
          catch (NumberFormatException e)
          {
            Log.w(TAG, "Failed to parse int-as-string for " + key + ": " + stringValue);
            return false;
          }
        }

        if (validateStringPreference(key, stringValue))
        {
          editor.putString(key, stringValue);
          return true;
        }
        else
        {
          Log.w(TAG, "Skipping invalid string value for " + key);
          return false;
        }
      }
    }
    else if (value.isJsonArray())
    {
      // Handle String Sets
      Set<String> stringSet = new HashSet<>();
      for (com.google.gson.JsonElement element : value.getAsJsonArray())
      {
        if (element.isJsonPrimitive() && element.getAsJsonPrimitive().isString())
        {
          stringSet.add(element.getAsString());
        }
      }
      editor.putStringSet(key, stringSet);
      return true;
    }

    Log.w(TAG, "Skipping unknown preference type for " + key);
    return false;
  }

  /**
   * Validate integer preference values
   */
  private boolean validateIntPreference(String key, int value)
  {
    switch (key)
    {
      // Opacity values (0-100)
      case "label_brightness":
      case "keyboard_opacity":
      case "key_opacity":
      case "key_activated_opacity":
      case "suggestion_bar_opacity":
        return value >= 0 && value <= 100;

      // Keyboard height percentages
      case "keyboard_height":
      case "keyboard_height_unfolded":
        return value >= 10 && value <= 100;
      case "keyboard_height_landscape":
      case "keyboard_height_landscape_unfolded":
        return value >= 20 && value <= 65;

      // Margins and spacing (0-200 dp max)
      case "margin_bottom_portrait":
      case "margin_bottom_landscape":
      case "margin_bottom_portrait_unfolded":
      case "margin_bottom_landscape_unfolded":
      case "horizontal_margin_portrait":
      case "horizontal_margin_landscape":
      case "horizontal_margin_portrait_unfolded":
      case "horizontal_margin_landscape_unfolded":
        return value >= 0 && value <= 200;

      // Border radius (0-100%)
      case "custom_border_radius":
        return value >= 0 && value <= 100;

      // Timing values (milliseconds)
      case "vibrate_duration":
        return value >= 0 && value <= 100;
      case "longpress_timeout":
        return value >= 50 && value <= 2000;
      case "longpress_interval":
        return value >= 5 && value <= 100;

      // Short gesture distance (10-95%)
      case "short_gesture_min_distance":
        return value >= 10 && value <= 95;

      // Neural network parameters
      case "neural_beam_width":
        return value >= 1 && value <= 16;
      case "neural_max_length":
        return value >= 10 && value <= 50;

      // Auto-correction parameters
      case "autocorrect_min_word_length":
        return value >= 2 && value <= 5;
      case "autocorrect_confidence_min_frequency":
        return value >= 100 && value <= 5000;

      // Clipboard history limit
      case "clipboard_history_limit":
        return value >= 1 && value <= 50;

      // Circle sensitivity
      case "circle_sensitivity":
        return value >= 1 && value <= 5;

      default:
        // Unknown integer preference - allow it (version-tolerant)
        return true;
    }
  }

  /**
   * Validate float preference values
   */
  private boolean validateFloatPreference(String key, float value)
  {
    switch (key)
    {
      // Character size (0.75-1.5)
      case "character_size":
        return value >= 0.75f && value <= 1.5f;

      // Margins (0-5%)
      case "key_vertical_margin":
      case "key_horizontal_margin":
        return value >= 0f && value <= 5f;

      // Border line width (0-5 dp)
      case "custom_border_line_width":
        return value >= 0f && value <= 5f;

      // Prediction weights
      case "prediction_context_boost":
        return value >= 0.5f && value <= 5.0f;
      case "prediction_frequency_scale":
        return value >= 100.0f && value <= 5000.0f;

      // Auto-correction threshold
      case "autocorrect_char_match_threshold":
        return value >= 0.5f && value <= 0.9f;

      // Neural confidence threshold
      case "neural_confidence_threshold":
        return value >= 0.0f && value <= 1.0f;

      default:
        // Unknown float preference - allow it (version-tolerant)
        return true;
    }
  }

  /**
   * Check if a preference stores data as a JSON string
   * These preferences use ListGroupPreference which stores data as JSON-encoded strings
   */
  private boolean isJsonStringPreference(String key)
  {
    switch (key)
    {
      // LayoutsPreference - stores List<Layout> as JSON string
      case "layouts":
      // ExtraKeysPreference - stores Map<KeyValue, PreferredPos> as JSON string
      case "extra_keys":
      // CustomExtraKeysPreference - stores Map<KeyValue, PreferredPos> as JSON string
      case "custom_extra_keys":
        return true;
      default:
        return false;
    }
  }

  /**
   * Check if a preference is internal state that shouldn't be exported/imported
   */
  private boolean isInternalPreference(String key)
  {
    switch (key)
    {
      // Internal version tracking
      case "version":
      // Current layout indices (managed by Config, device-specific)
      case "current_layout_portrait":
      case "current_layout_landscape":
        return true;
      default:
        return false;
    }
  }

  /**
   * Check if a preference is stored as a float in SharedPreferences
   * This is critical because SharedPreferences throws ClassCastException if you
   * try to read an int as float or vice versa
   */
  private boolean isFloatPreference(String key)
  {
    switch (key)
    {
      // Character and UI sizing
      case "character_size":
      case "key_vertical_margin":
      case "key_horizontal_margin":
      case "custom_border_line_width":

      // Prediction weights
      case "prediction_context_boost":
      case "prediction_frequency_scale":

      // Auto-correction threshold
      case "autocorrect_char_match_threshold":

      // Neural confidence threshold
      case "neural_confidence_threshold":
        return true;
      default:
        return false;
    }
  }

  /**
   * Check if a preference stores integers as strings (from ListPreference)
   * These need to be parsed and stored as int to prevent ClassCastException
   */
  private boolean isIntegerStoredAsString(String key)
  {
    switch (key)
    {
      // ListPreference values that are actually integers
      case "circle_sensitivity":
      case "show_numpad":
      case "clipboard_history_limit":
        return true;
      default:
        return false;
    }
  }

  /**
   * Validate string preference values
   */
  private boolean validateStringPreference(String key, String value)
  {
    if (value == null)
      return false;

    switch (key)
    {
      // Theme values - relaxed validation for forward compatibility
      // New themes added in future versions should still import successfully
      case "theme":
        // Just ensure it's not empty - app will fall back to default if invalid
        return !value.isEmpty();

      // Number row options
      case "number_row":
        return value.matches("no_number_row|no_symbols|symbols");

      // Show numpad options
      case "show_numpad":
        return value.matches("never|always|landscape|[0-9]+");

      // Numpad layout
      case "numpad_layout":
        return value.matches("high_first|low_first|default");

      // Number entry layout
      case "number_entry_layout":
        return value.matches("pin|number");

      // Circle sensitivity (string representation)
      case "circle_sensitivity":
        return value.matches("[1-5]");

      // Slider sensitivity (string representation)
      case "slider_sensitivity":
        return value.matches("[0-9]+");

      // Swipe distance (string representation)
      case "swipe_dist":
        return value.matches("[0-9]+(\\.[0-9]+)?");

      default:
        // Unknown string preference - allow it (version-tolerant)
        return true;
    }
  }

  /**
   * Result of import operation
   */
  public static class ImportResult
  {
    public int importedCount = 0;
    public int skippedCount = 0;
    public String sourceVersion = "unknown";
    public int sourceScreenWidth = 0;
    public int sourceScreenHeight = 0;
    public int currentScreenWidth = 0;
    public int currentScreenHeight = 0;
    public Set<String> importedKeys = new HashSet<>();
    public Set<String> skippedKeys = new HashSet<>();

    public boolean hasScreenSizeMismatch()
    {
      if (sourceScreenWidth == 0 || sourceScreenHeight == 0)
        return false; // No source dimensions available

      int widthDiff = Math.abs(currentScreenWidth - sourceScreenWidth);
      int heightDiff = Math.abs(currentScreenHeight - sourceScreenHeight);

      // Consider it a mismatch if either dimension differs by more than 20%
      return (widthDiff > currentScreenWidth * 0.2) ||
             (heightDiff > currentScreenHeight * 0.2);
    }
  }
}
