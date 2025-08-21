package juloo.keyboard2.ml;

import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Persistent storage manager for ML swipe data.
 * Uses SQLite for efficient storage and querying, with JSON export capability.
 */
public class SwipeMLDataStore extends SQLiteOpenHelper
{
  private static final String TAG = "SwipeMLDataStore";
  
  // Database configuration
  private static final String DATABASE_NAME = "swipe_ml_data.db";
  private static final int DATABASE_VERSION = 1;
  
  // Table and column names
  private static final String TABLE_SWIPES = "swipe_data";
  private static final String COL_ID = "id";
  private static final String COL_TRACE_ID = "trace_id";
  private static final String COL_TARGET_WORD = "target_word";
  private static final String COL_TIMESTAMP = "timestamp_utc";
  private static final String COL_SOURCE = "collection_source";
  private static final String COL_JSON_DATA = "json_data";
  private static final String COL_IS_EXPORTED = "is_exported";
  
  // Statistics tracking
  private static final String PREF_NAME = "swipe_ml_stats";
  private static final String PREF_TOTAL_COUNT = "total_swipes";
  private static final String PREF_CALIBRATION_COUNT = "calibration_swipes";
  private static final String PREF_USER_COUNT = "user_swipes";
  
  private final Context _context;
  private final ExecutorService _executor;
  private static SwipeMLDataStore _instance;
  
  /**
   * Get singleton instance
   */
  public static synchronized SwipeMLDataStore getInstance(Context context)
  {
    if (_instance == null)
    {
      _instance = new SwipeMLDataStore(context.getApplicationContext());
    }
    return _instance;
  }
  
  private SwipeMLDataStore(Context context)
  {
    super(context, DATABASE_NAME, null, DATABASE_VERSION);
    _context = context;
    _executor = Executors.newSingleThreadExecutor();
  }
  
  @Override
  public void onCreate(SQLiteDatabase db)
  {
    // Create main data table
    String createTable = "CREATE TABLE " + TABLE_SWIPES + " (" +
      COL_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
      COL_TRACE_ID + " TEXT UNIQUE NOT NULL, " +
      COL_TARGET_WORD + " TEXT NOT NULL, " +
      COL_TIMESTAMP + " INTEGER NOT NULL, " +
      COL_SOURCE + " TEXT NOT NULL, " +
      COL_JSON_DATA + " TEXT NOT NULL, " +
      COL_IS_EXPORTED + " INTEGER DEFAULT 0, " +
      "INDEX idx_word (target_word), " +
      "INDEX idx_source (collection_source), " +
      "INDEX idx_timestamp (timestamp_utc)" +
      ")";
    db.execSQL(createTable);
    
    Log.d(TAG, "Database created with version " + DATABASE_VERSION);
  }
  
  @Override
  public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion)
  {
    // Handle future schema upgrades
    Log.w(TAG, "Upgrading database from version " + oldVersion + " to " + newVersion);
  }
  
  /**
   * Store swipe data asynchronously
   */
  public void storeSwipeData(final SwipeMLData data)
  {
    if (!data.isValid())
    {
      Log.w(TAG, "Invalid swipe data, not storing");
      return;
    }
    
    _executor.execute(() -> {
      try
      {
        SQLiteDatabase db = getWritableDatabase();
        
        // Check if trace_id already exists
        Cursor cursor = db.query(TABLE_SWIPES, new String[]{COL_ID},
          COL_TRACE_ID + "=?", new String[]{data.getTraceId()},
          null, null, null);
        
        if (cursor.getCount() > 0)
        {
          cursor.close();
          Log.w(TAG, "Trace ID already exists: " + data.getTraceId());
          return;
        }
        cursor.close();
        
        // Store the data
        android.content.ContentValues values = new android.content.ContentValues();
        values.put(COL_TRACE_ID, data.getTraceId());
        values.put(COL_TARGET_WORD, data.getTargetWord());
        values.put(COL_TIMESTAMP, data.getTimestampUtc());
        values.put(COL_SOURCE, data.getCollectionSource());
        values.put(COL_JSON_DATA, data.toJSON().toString());
        values.put(COL_IS_EXPORTED, 0);
        
        long rowId = db.insert(TABLE_SWIPES, null, values);
        
        if (rowId != -1)
        {
          Log.d(TAG, "Stored swipe data: " + data.getTargetWord() + 
                     " (" + data.getCollectionSource() + ")");
          updateStatistics(data.getCollectionSource());
        }
        else
        {
          Log.e(TAG, "Failed to store swipe data");
        }
      }
      catch (Exception e)
      {
        Log.e(TAG, "Error storing swipe data", e);
      }
    });
  }
  
  /**
   * Store multiple swipe data entries (batch operation)
   */
  public void storeSwipeDataBatch(final List<SwipeMLData> dataList)
  {
    _executor.execute(() -> {
      SQLiteDatabase db = getWritableDatabase();
      db.beginTransaction();
      try
      {
        for (SwipeMLData data : dataList)
        {
          if (!data.isValid())
            continue;
          
          android.content.ContentValues values = new android.content.ContentValues();
          values.put(COL_TRACE_ID, data.getTraceId());
          values.put(COL_TARGET_WORD, data.getTargetWord());
          values.put(COL_TIMESTAMP, data.getTimestampUtc());
          values.put(COL_SOURCE, data.getCollectionSource());
          values.put(COL_JSON_DATA, data.toJSON().toString());
          values.put(COL_IS_EXPORTED, 0);
          
          db.insertWithOnConflict(TABLE_SWIPES, null, values,
                                  SQLiteDatabase.CONFLICT_IGNORE);
        }
        db.setTransactionSuccessful();
        Log.d(TAG, "Stored batch of " + dataList.size() + " swipe entries");
      }
      catch (Exception e)
      {
        Log.e(TAG, "Error storing batch data", e);
      }
      finally
      {
        db.endTransaction();
      }
    });
  }
  
  /**
   * Load all swipe data (for export or training)
   */
  public List<SwipeMLData> loadAllData()
  {
    List<SwipeMLData> dataList = new ArrayList<>();
    SQLiteDatabase db = getReadableDatabase();
    
    Cursor cursor = db.query(TABLE_SWIPES, new String[]{COL_JSON_DATA},
      null, null, null, null, COL_TIMESTAMP + " ASC");
    
    while (cursor.moveToNext())
    {
      try
      {
        String jsonStr = cursor.getString(0);
        JSONObject json = new JSONObject(jsonStr);
        dataList.add(new SwipeMLData(json));
      }
      catch (JSONException e)
      {
        Log.e(TAG, "Error parsing stored JSON", e);
      }
    }
    cursor.close();
    
    Log.d(TAG, "Loaded " + dataList.size() + " swipe entries");
    return dataList;
  }
  
  /**
   * Load data by collection source
   */
  public List<SwipeMLData> loadDataBySource(String source)
  {
    List<SwipeMLData> dataList = new ArrayList<>();
    SQLiteDatabase db = getReadableDatabase();
    
    Cursor cursor = db.query(TABLE_SWIPES, new String[]{COL_JSON_DATA},
      COL_SOURCE + "=?", new String[]{source},
      null, null, COL_TIMESTAMP + " ASC");
    
    while (cursor.moveToNext())
    {
      try
      {
        String jsonStr = cursor.getString(0);
        JSONObject json = new JSONObject(jsonStr);
        dataList.add(new SwipeMLData(json));
      }
      catch (JSONException e)
      {
        Log.e(TAG, "Error parsing stored JSON", e);
      }
    }
    cursor.close();
    
    return dataList;
  }
  
  /**
   * Load recent data for incremental training
   */
  public List<SwipeMLData> loadRecentData(int limit)
  {
    List<SwipeMLData> dataList = new ArrayList<>();
    SQLiteDatabase db = getReadableDatabase();
    
    Cursor cursor = db.query(TABLE_SWIPES, new String[]{COL_JSON_DATA},
      null, null, null, null, 
      COL_TIMESTAMP + " DESC", String.valueOf(limit));
    
    while (cursor.moveToNext())
    {
      try
      {
        String jsonStr = cursor.getString(0);
        JSONObject json = new JSONObject(jsonStr);
        dataList.add(new SwipeMLData(json));
      }
      catch (JSONException e)
      {
        Log.e(TAG, "Error parsing stored JSON", e);
      }
    }
    cursor.close();
    
    return dataList;
  }
  
  /**
   * Export all data to JSON file
   */
  public File exportToJSON() throws IOException, JSONException
  {
    List<SwipeMLData> allData = loadAllData();
    
    // Create export directory
    File exportDir = new File(_context.getExternalFilesDir(null), "swipe_ml_export");
    if (!exportDir.exists())
    {
      exportDir.mkdirs();
    }
    
    // Create filename with timestamp
    SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US);
    String filename = "swipe_data_" + sdf.format(new Date()) + ".json";
    File exportFile = new File(exportDir, filename);
    
    // Build JSON array
    JSONArray jsonArray = new JSONArray();
    for (SwipeMLData data : allData)
    {
      jsonArray.put(data.toJSON());
    }
    
    // Add metadata
    JSONObject root = new JSONObject();
    root.put("export_version", "1.0");
    root.put("export_timestamp", System.currentTimeMillis());
    root.put("total_samples", allData.size());
    root.put("database_version", DATABASE_VERSION);
    root.put("data", jsonArray);
    
    // Add statistics
    SharedPreferences prefs = _context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    JSONObject stats = new JSONObject();
    stats.put("total_swipes", prefs.getInt(PREF_TOTAL_COUNT, 0));
    stats.put("calibration_swipes", prefs.getInt(PREF_CALIBRATION_COUNT, 0));
    stats.put("user_swipes", prefs.getInt(PREF_USER_COUNT, 0));
    root.put("statistics", stats);
    
    // Write to file
    FileWriter writer = new FileWriter(exportFile);
    writer.write(root.toString(2)); // Pretty print with 2-space indent
    writer.close();
    
    // Mark all as exported
    markAllAsExported();
    
    Log.i(TAG, "Exported " + allData.size() + " entries to " + exportFile.getAbsolutePath());
    return exportFile;
  }
  
  /**
   * Export to newline-delimited JSON (NDJSON) for streaming processing
   */
  public File exportToNDJSON() throws IOException, JSONException
  {
    List<SwipeMLData> allData = loadAllData();
    
    File exportDir = new File(_context.getExternalFilesDir(null), "swipe_ml_export");
    if (!exportDir.exists())
    {
      exportDir.mkdirs();
    }
    
    SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US);
    String filename = "swipe_data_" + sdf.format(new Date()) + ".ndjson";
    File exportFile = new File(exportDir, filename);
    
    FileWriter writer = new FileWriter(exportFile);
    for (SwipeMLData data : allData)
    {
      writer.write(data.toJSON().toString());
      writer.write("\n");
    }
    writer.close();
    
    Log.i(TAG, "Exported " + allData.size() + " entries to NDJSON: " + exportFile.getAbsolutePath());
    return exportFile;
  }
  
  /**
   * Get statistics about stored data
   */
  public DataStatistics getStatistics()
  {
    SQLiteDatabase db = getReadableDatabase();
    DataStatistics stats = new DataStatistics();
    
    // Total count
    Cursor cursor = db.rawQuery("SELECT COUNT(*) FROM " + TABLE_SWIPES, null);
    if (cursor.moveToFirst())
    {
      stats.totalCount = cursor.getInt(0);
    }
    cursor.close();
    
    // Count by source
    cursor = db.rawQuery("SELECT " + COL_SOURCE + ", COUNT(*) FROM " + TABLE_SWIPES + 
                         " GROUP BY " + COL_SOURCE, null);
    while (cursor.moveToNext())
    {
      String source = cursor.getString(0);
      int count = cursor.getInt(1);
      if ("calibration".equals(source))
      {
        stats.calibrationCount = count;
      }
      else if ("user_selection".equals(source))
      {
        stats.userSelectionCount = count;
      }
    }
    cursor.close();
    
    // Unique words
    cursor = db.rawQuery("SELECT COUNT(DISTINCT " + COL_TARGET_WORD + ") FROM " + TABLE_SWIPES, null);
    if (cursor.moveToFirst())
    {
      stats.uniqueWords = cursor.getInt(0);
    }
    cursor.close();
    
    return stats;
  }
  
  /**
   * Clear all data (with confirmation)
   */
  public void clearAllData()
  {
    SQLiteDatabase db = getWritableDatabase();
    db.delete(TABLE_SWIPES, null, null);
    
    // Reset statistics
    SharedPreferences prefs = _context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    prefs.edit().clear().apply();
    
    Log.w(TAG, "All swipe data cleared");
  }
  
  /**
   * Mark all entries as exported
   */
  private void markAllAsExported()
  {
    SQLiteDatabase db = getWritableDatabase();
    android.content.ContentValues values = new android.content.ContentValues();
    values.put(COL_IS_EXPORTED, 1);
    db.update(TABLE_SWIPES, values, null, null);
  }
  
  /**
   * Update statistics
   */
  private void updateStatistics(String source)
  {
    SharedPreferences prefs = _context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = prefs.edit();
    
    // Increment total count
    int total = prefs.getInt(PREF_TOTAL_COUNT, 0);
    editor.putInt(PREF_TOTAL_COUNT, total + 1);
    
    // Increment source-specific count
    if ("calibration".equals(source))
    {
      int count = prefs.getInt(PREF_CALIBRATION_COUNT, 0);
      editor.putInt(PREF_CALIBRATION_COUNT, count + 1);
    }
    else if ("user_selection".equals(source))
    {
      int count = prefs.getInt(PREF_USER_COUNT, 0);
      editor.putInt(PREF_USER_COUNT, count + 1);
    }
    
    editor.apply();
  }
  
  /**
   * Statistics class
   */
  public static class DataStatistics
  {
    public int totalCount;
    public int calibrationCount;
    public int userSelectionCount;
    public int uniqueWords;
    
    @Override
    public String toString()
    {
      return String.format(Locale.US,
        "Total: %d, Calibration: %d, User: %d, Unique words: %d",
        totalCount, calibrationCount, userSelectionCount, uniqueWords);
    }
  }
}