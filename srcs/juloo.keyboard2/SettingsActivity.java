package juloo.keyboard2;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.preference.Preference;
import android.preference.PreferenceActivity;
import android.preference.PreferenceManager;
import android.util.Log;
import android.widget.Toast;
import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Properties;
import juloo.keyboard2.ml.SwipeMLDataStore;
import juloo.keyboard2.ml.SwipeMLTrainer;
import android.app.ProgressDialog;

public class SettingsActivity extends PreferenceActivity
  implements SharedPreferences.OnSharedPreferenceChangeListener
{
  // Request codes for backup/restore file picker
  private static final int REQUEST_CODE_BACKUP = 1001;
  private static final int REQUEST_CODE_RESTORE = 1002;
  private static final int REQUEST_CODE_NEURAL_ENCODER = 1003;
  private static final int REQUEST_CODE_NEURAL_DECODER = 1004;

  private BackupRestoreManager backupRestoreManager;

  @Override
  public void onCreate(Bundle savedInstanceState)
  {
    super.onCreate(savedInstanceState);
    // The preferences can't be read when in direct-boot mode. Avoid crashing
    // and don't allow changing the settings.
    // Run the config migration on this prefs as it might be different from the
    // one used by the keyboard, which have been migrated.
    try
    {
      Config.migrate(getPreferenceManager().getSharedPreferences());
    }
    catch (Exception _e) { fallbackEncrypted(); return; }
    addPreferencesFromResource(R.xml.settings);

    // Initialize backup/restore manager
    backupRestoreManager = new BackupRestoreManager(this);

    // Setup backup/restore preference handlers
    setupBackupRestoreHandlers();

    // Add CGR reset button handlers and update summaries
    setupCGRResetButtons();
    updateCGRParameterSummaries();

    boolean foldableDevice = FoldStateTracker.isFoldableDevice(this);
    findPreference("margin_bottom_portrait_unfolded").setEnabled(foldableDevice);
    findPreference("margin_bottom_landscape_unfolded").setEnabled(foldableDevice);
    findPreference("horizontal_margin_portrait_unfolded").setEnabled(foldableDevice);
    findPreference("horizontal_margin_landscape_unfolded").setEnabled(foldableDevice);
    findPreference("keyboard_height_unfolded").setEnabled(foldableDevice);
    findPreference("keyboard_height_landscape_unfolded").setEnabled(foldableDevice);
    
    // Add version info display
    Preference versionPref = findPreference("version_info");
    if (versionPref != null)
    {
      try
      {
        Properties versionInfo = loadVersionInfo();
        String commit = versionInfo.getProperty("commit", "unknown");
        String commitDate = versionInfo.getProperty("commit_date", "");
        String buildDate = versionInfo.getProperty("build_date", "");
        String buildNumber = versionInfo.getProperty("build_number", "");
        
        versionPref.setTitle("Version Info");
        versionPref.setSummary(String.format("Build: %s\nCommit: %s (%s)\nBuilt: %s",
          buildNumber.substring(Math.max(0, buildNumber.length() - 8)),
          commit, commitDate, buildDate));
      }
      catch (Exception e)
      {
        versionPref.setSummary("Version info unavailable");
        android.util.Log.e("SettingsActivity", "Failed to load version info", e);
      }
    }
    
    // Add update button
    Preference updatePref = findPreference("update_app");
    if (updatePref != null)
    {
      updatePref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          installUpdate();
          return true;
        }
      });
    }
    
    // Set up calibration preference click handler
    Preference calibrationPref = findPreference("swipe_calibration");
    if (calibrationPref != null)
    {
      calibrationPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          Intent intent = new Intent(SettingsActivity.this, SwipeCalibrationActivity.class);
          startActivity(intent);
          return true;
        }
      });
    }

    // Set up debug preference click handler
    Preference debugPref = findPreference("swipe_debug");
    if (debugPref != null)
    {
      debugPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          Intent intent = new Intent(SettingsActivity.this, SwipeDebugActivity.class);
          startActivity(intent);
          return true;
        }
      });
    }

    // Set up dictionary manager preference click handler
    Preference dictionaryManagerPref = findPreference("dictionary_manager");
    if (dictionaryManagerPref != null)
    {
      dictionaryManagerPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          Intent intent = new Intent(SettingsActivity.this, DictionaryManagerActivity.class);
          startActivity(intent);
          return true;
        }
      });
    }

    // Set up ML data export preference (PreferenceScreen type)
    Preference exportMLDataPref = findPreference("export_swipe_ml_data");
    if (exportMLDataPref != null)
    {
      try
      {
        // Update summary with current data statistics
        SwipeMLDataStore dataStore = SwipeMLDataStore.getInstance(this);
        SwipeMLDataStore.DataStatistics stats = dataStore.getStatistics();
        exportMLDataPref.setSummary("Export all collected swipe data (" + stats.totalCount + " samples)");
      }
      catch (Exception e)
      {
        exportMLDataPref.setSummary("Export all collected swipe data");
        android.util.Log.e("SettingsActivity", "Failed to get ML data statistics", e);
      }
      
      exportMLDataPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          exportSwipeMLData();
          return true;
        }
      });
    }
    
    // Import ML Data
    Preference importMLDataPref = findPreference("import_swipe_ml_data");
    if (importMLDataPref != null)
    {
      importMLDataPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          importSwipeMLData();
          return true;
        }
      });
    }
    
    // Set up ML training preference (PreferenceScreen type)
    Preference trainMLModelPref = findPreference("train_swipe_ml_model");
    if (trainMLModelPref != null)
    {
      try
      {
        SwipeMLDataStore dataStore = SwipeMLDataStore.getInstance(this);
        SwipeMLDataStore.DataStatistics stats = dataStore.getStatistics();
        trainMLModelPref.setSummary("Train model with " + stats.totalCount + " samples (min 100 required)");
      }
      catch (Exception e)
      {
        trainMLModelPref.setSummary("Train swipe prediction model");
        android.util.Log.e("SettingsActivity", "Failed to get ML data statistics", e);
      }
      
      trainMLModelPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          startMLTraining();
          return true;
        }
      });
    }

    // Set up neural model file pickers
    Preference loadEncoderPref = findPreference("neural_load_encoder");
    if (loadEncoderPref != null)
    {
      loadEncoderPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          openFilePicker(REQUEST_CODE_NEURAL_ENCODER);
          return true;
        }
      });
    }

    Preference loadDecoderPref = findPreference("neural_load_decoder");
    if (loadDecoderPref != null)
    {
      loadDecoderPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          openFilePicker(REQUEST_CODE_NEURAL_DECODER);
          return true;
        }
      });
    }

    // Update neural model info display
    updateNeuralModelInfo();
  }

  private void updateNeuralModelInfo()
  {
    Preference modelInfoPref = findPreference("neural_model_info");
    if (modelInfoPref != null)
    {
      try
      {
        // Get model info from singleton predictor
        OnnxSwipePredictor predictor = OnnxSwipePredictor.getInstance(this);
        if (predictor != null && predictor.isAvailable())
        {
          String modelInfo = predictor.getModelInfo();
          modelInfoPref.setSummary("✅ Loaded: " + modelInfo);
        }
        else
        {
          modelInfoPref.setSummary("⚠️ Model not loaded");
        }
      }
      catch (Exception e)
      {
        modelInfoPref.setSummary("❌ Error loading model info");
        android.util.Log.e("SettingsActivity", "Failed to get model info", e);
      }
    }
  }

  private void exportSwipeMLData()
  {
    try
    {
      SwipeMLDataStore dataStore = SwipeMLDataStore.getInstance(this);
      SwipeMLDataStore.DataStatistics stats = dataStore.getStatistics();
      
      if (stats.totalCount == 0)
      {
        Toast.makeText(this, "No swipe data to export", Toast.LENGTH_SHORT).show();
        return;
      }
      
      // Export to JSON file
      File exportFile = dataStore.exportToJSON();
      
      // Show success message with file location
      String message = "Exported " + stats.totalCount + " swipe samples\n\n" +
                      "File saved to:\n" + exportFile.getAbsolutePath() + "\n\n" +
                      "Statistics:\n" +
                      "• Calibration samples: " + stats.calibrationCount + "\n" +
                      "• User samples: " + stats.userSelectionCount + "\n" +
                      "• Unique words: " + stats.uniqueWords;
      
      // Create alert dialog to show export info
      android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(this);
      builder.setTitle("Export Successful");
      builder.setMessage(message);
      builder.setPositiveButton("OK", null);
      
      // Add copy path button
      builder.setNeutralButton("Copy Path", (dialog, which) -> {
        android.content.ClipboardManager clipboard = 
          (android.content.ClipboardManager) getSystemService(android.content.Context.CLIPBOARD_SERVICE);
        android.content.ClipData clip = android.content.ClipData.newPlainText("Export Path", exportFile.getAbsolutePath());
        clipboard.setPrimaryClip(clip);
        Toast.makeText(this, "Path copied to clipboard", Toast.LENGTH_SHORT).show();
      });
      
      builder.show();
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Export failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
      android.util.Log.e("SettingsActivity", "Failed to export ML data", e);
    }
  }
  
  private void importSwipeMLData()
  {
    // Create file picker dialog
    android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(this);
    builder.setTitle("Import Swipe Data");
    builder.setMessage("Select JSON file to import from:");
    
    // Check common locations for exported files
    File[] possibleFiles = new File[] {
      new File("/sdcard/Android/data/juloo.keyboard2.debug/files/swipe_ml_export/"),
      new File("/storage/emulated/0/Android/data/juloo.keyboard2.debug/files/swipe_ml_export/"),
      new File("/sdcard/Download/"),
      new File("/storage/emulated/0/Download/")
    };
    
    // Find existing JSON files
    java.util.List<File> jsonFiles = new java.util.ArrayList<>();
    for (File dir : possibleFiles)
    {
      if (dir.exists() && dir.isDirectory())
      {
        File[] files = dir.listFiles((dir1, name) -> name.endsWith(".json"));
        if (files != null)
        {
          for (File f : files)
          {
            jsonFiles.add(f);
          }
        }
      }
    }
    
    if (jsonFiles.isEmpty())
    {
      Toast.makeText(this, "No JSON files found in common locations", Toast.LENGTH_LONG).show();
      return;
    }
    
    // Create list of file names
    String[] fileNames = new String[jsonFiles.size()];
    for (int i = 0; i < jsonFiles.size(); i++)
    {
      fileNames[i] = jsonFiles.get(i).getName() + "\n(" + jsonFiles.get(i).getParent() + ")";
    }
    
    builder.setItems(fileNames, (dialog, which) -> {
      File selectedFile = jsonFiles.get(which);
      performImport(selectedFile);
    });
    
    builder.setNegativeButton("Cancel", null);
    builder.show();
  }
  
  private void performImport(File jsonFile)
  {
    try
    {
      SwipeMLDataStore dataStore = SwipeMLDataStore.getInstance(this);
      int importedCount = dataStore.importFromJSON(jsonFile);
      
      if (importedCount > 0)
      {
        Toast.makeText(this, "Successfully imported " + importedCount + " swipe samples", 
                      Toast.LENGTH_LONG).show();
        
        // Update export preference summary
        Preference exportPref = findPreference("export_swipe_ml_data");
        if (exportPref != null)
        {
          SwipeMLDataStore.DataStatistics stats = dataStore.getStatistics();
          exportPref.setSummary("Export all collected swipe data (" + stats.totalCount + " samples)");
        }
      }
      else
      {
        Toast.makeText(this, "No new samples imported (duplicates skipped)", Toast.LENGTH_SHORT).show();
      }
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Import failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
      android.util.Log.e("SettingsActivity", "Failed to import ML data", e);
    }
  }
  
  private void startMLTraining()
  {
    SwipeMLTrainer trainer = new SwipeMLTrainer(this);
    
    if (!trainer.canTrain())
    {
      Toast.makeText(this, "Not enough data for training. Need at least 100 samples.", 
                     Toast.LENGTH_LONG).show();
      return;
    }
    
    // Create progress dialog
    final ProgressDialog progressDialog = new ProgressDialog(this);
    progressDialog.setTitle("Training ML Model");
    progressDialog.setMessage("Preparing training data...");
    progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
    progressDialog.setMax(100);
    progressDialog.setCancelable(false);
    progressDialog.show();
    
    // Set up training listener
    trainer.setTrainingListener(new SwipeMLTrainer.TrainingListener()
    {
      @Override
      public void onTrainingStarted()
      {
        runOnUiThread(() -> {
          progressDialog.setMessage("Training in progress...");
        });
      }
      
      @Override
      public void onTrainingProgress(int progress, int total)
      {
        runOnUiThread(() -> {
          progressDialog.setProgress(progress);
        });
      }
      
      @Override
      public void onTrainingCompleted(SwipeMLTrainer.TrainingResult result)
      {
        runOnUiThread(() -> {
          progressDialog.dismiss();
          String message = String.format(
            "Training completed!\nSamples: %d\nTime: %.1f seconds\nAccuracy: %.1f%%",
            result.samplesUsed, result.trainingTimeMs / 1000.0, result.accuracy * 100);
          Toast.makeText(SettingsActivity.this, message, Toast.LENGTH_LONG).show();
        });
      }
      
      @Override
      public void onTrainingError(String error)
      {
        runOnUiThread(() -> {
          progressDialog.dismiss();
          Toast.makeText(SettingsActivity.this, "Training failed: " + error, 
                         Toast.LENGTH_LONG).show();
        });
      }
    });
    
    // Start training
    trainer.startTraining();
  }
  
  private Properties loadVersionInfo()
  {
    Properties props = new Properties();
    try
    {
      BufferedReader reader = new BufferedReader(
        new InputStreamReader(getResources().openRawResource(
          getResources().getIdentifier("version_info", "raw", getPackageName()))));
      props.load(reader);
      reader.close();
    }
    catch (Exception e)
    {
      android.util.Log.e("SettingsActivity", "Failed to load version info", e);
    }
    return props;
  }
  
  private void installUpdate()
  {
    // Check multiple possible APK locations
    File[] possibleLocations = new File[] {
      new File("/sdcard/unexpected/debug-kb.apk"),
      new File("/storage/emulated/0/unexpected/debug-kb.apk"),
      new File("/sdcard/Download/juloo.keyboard2.debug.apk"),
      new File("/storage/emulated/0/Download/juloo.keyboard2.debug.apk"),
      new File(android.os.Environment.getExternalStoragePublicDirectory(
               android.os.Environment.DIRECTORY_DOWNLOADS), "juloo.keyboard2.debug.apk"),
      // Also check the build output location if accessible
      new File("/data/data/com.termux/files/home/git/Unexpected-Keyboard/build/outputs/apk/debug/juloo.keyboard2.debug.apk")
    };
    
    File updateApk = null;
    for (File location : possibleLocations)
    {
      if (location.exists() && location.canRead())
      {
        updateApk = location;
        break;
      }
    }
    
    if (updateApk == null)
    {
      // Show helpful dialog about where to place APK
      android.app.AlertDialog.Builder helpBuilder = new android.app.AlertDialog.Builder(this);
      helpBuilder.setTitle("No Update APK Found");
      helpBuilder.setMessage("Please place the APK file in one of these locations:\n\n" +
                           "• /sdcard/Download/juloo.keyboard2.debug.apk\n" +
                           "• /sdcard/unexpected/debug-kb.apk\n\n" +
                           "Or build it with:\n" +
                           "./gradlew assembleDebug");
      helpBuilder.setPositiveButton("OK", null);
      helpBuilder.show();
      return;
    }
    
    final File apkFile = updateApk;
    
    // Show dialog with working installation methods
    android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(this);
    builder.setTitle("Install Update");
    builder.setMessage("APK found at:\n" + apkFile.getName() + "\n\nSize: " + 
                       (apkFile.length() / 1024) + " KB");
    
    // Option 1: Use Android's package installer (most reliable)
    builder.setPositiveButton("Install Now", (dialog, which) -> {
      try
      {
        // Use FileProvider for Android 7.0+ (API 24+)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N)
        {
          // Try to use content URI with FileProvider
          Intent intent = new Intent(Intent.ACTION_INSTALL_PACKAGE);
          intent.setDataAndType(android.net.Uri.fromFile(apkFile), 
                               "application/vnd.android.package-archive");
          intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_GRANT_READ_URI_PERMISSION);
          startActivity(intent);
        }
        else
        {
          // For older Android versions
          Intent intent = new Intent(Intent.ACTION_VIEW);
          intent.setDataAndType(android.net.Uri.fromFile(apkFile),
                               "application/vnd.android.package-archive");
          intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
          startActivity(intent);
        }
      }
      catch (Exception e)
      {
        // If direct install fails, try alternative method
        try
        {
          Intent intent = new Intent(Intent.ACTION_VIEW);
          intent.setDataAndType(android.net.Uri.parse("file://" + apkFile.getAbsolutePath()),
                               "application/vnd.android.package-archive");
          intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
          startActivity(intent);
        }
        catch (Exception e2)
        {
          Toast.makeText(this, "Install failed: " + e2.getMessage() + 
                        "\n\nTry installing manually from file manager", Toast.LENGTH_LONG).show();
        }
      }
    });
    
    // Option 2: Copy to clipboard for manual install
    builder.setNeutralButton("Copy Path", (dialog, which) -> {
      android.content.ClipboardManager clipboard = 
        (android.content.ClipboardManager) getSystemService(android.content.Context.CLIPBOARD_SERVICE);
      android.content.ClipData clip = android.content.ClipData.newPlainText("APK Path", 
                                                                           apkFile.getAbsolutePath());
      clipboard.setPrimaryClip(clip);
      Toast.makeText(this, "Path copied!\nUse a file manager to navigate to:\n" + 
                    apkFile.getParent(), Toast.LENGTH_LONG).show();
    });
    
    // Option 3: Share APK to other apps
    builder.setNegativeButton("Share", (dialog, which) -> {
      try
      {
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("application/vnd.android.package-archive");
        shareIntent.putExtra(Intent.EXTRA_STREAM, android.net.Uri.fromFile(apkFile));
        shareIntent.putExtra(Intent.EXTRA_TEXT, "Install Unexpected Keyboard update");
        startActivity(Intent.createChooser(shareIntent, "Share APK via"));
      }
      catch (Exception e)
      {
        Toast.makeText(this, "Share failed: " + e.getMessage(), Toast.LENGTH_SHORT).show();
      }
    });
    
    builder.show();
  }
  
  void fallbackEncrypted()
  {
    // Can't communicate with the user here.
    finish();
  }

  protected void onStop()
  {
    DirectBootAwarePreferences
      .copy_preferences_to_protected_storage(this,
          getPreferenceManager().getSharedPreferences());
    super.onStop();
  }
  
  /**
   * Setup backup/restore preference handlers
   */
  private void setupBackupRestoreHandlers()
  {
    // Backup configuration
    Preference backupPref = findPreference("backup_config");
    if (backupPref != null)
    {
      backupPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          startBackup();
          return true;
        }
      });
    }

    // Restore configuration
    Preference restorePref = findPreference("restore_config");
    if (restorePref != null)
    {
      restorePref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          startRestore();
          return true;
        }
      });
    }
  }

  /**
   * Start backup process using Storage Access Framework
   */
  private void startBackup()
  {
    // Create filename with timestamp
    String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
    String fileName = "kb-config-" + timestamp + ".json";

    // Use Storage Access Framework to let user choose location
    Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
    intent.addCategory(Intent.CATEGORY_OPENABLE);
    intent.setType("application/json");
    intent.putExtra(Intent.EXTRA_TITLE, fileName);

    try
    {
      startActivityForResult(intent, REQUEST_CODE_BACKUP);
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Failed to open file picker: " + e.getMessage(),
                     Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Failed to start backup", e);
    }
  }

  /**
   * Start restore process using Storage Access Framework
   */
  private void startRestore()
  {
    // Use Storage Access Framework to let user choose file
    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
    intent.addCategory(Intent.CATEGORY_OPENABLE);
    intent.setType("application/json");

    try
    {
      startActivityForResult(intent, REQUEST_CODE_RESTORE);
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Failed to open file picker: " + e.getMessage(),
                     Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Failed to start restore", e);
    }
  }

  /**
   * Open file picker for ONNX model selection
   */
  private void openFilePicker(int requestCode)
  {
    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
    intent.addCategory(Intent.CATEGORY_OPENABLE);
    intent.setType("*/*"); // ONNX files might not have MIME type registered

    try
    {
      startActivityForResult(intent, requestCode);
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Failed to open file picker: " + e.getMessage(),
                     Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Failed to start file picker", e);
    }
  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data)
  {
    super.onActivityResult(requestCode, resultCode, data);

    if (resultCode != RESULT_OK || data == null)
    {
      return;
    }

    Uri uri = data.getData();
    if (uri == null)
    {
      return;
    }

    if (requestCode == REQUEST_CODE_BACKUP)
    {
      performBackup(uri);
    }
    else if (requestCode == REQUEST_CODE_RESTORE)
    {
      performRestore(uri);
    }
    else if (requestCode == REQUEST_CODE_NEURAL_ENCODER)
    {
      handleNeuralModelFile(uri, true);
    }
    else if (requestCode == REQUEST_CODE_NEURAL_DECODER)
    {
      handleNeuralModelFile(uri, false);
    }
  }

  private void handleNeuralModelFile(Uri uri, boolean isEncoder)
  {
    try
    {
      // Take persistent permission to access the file
      final int takeFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION;
      getContentResolver().takePersistableUriPermission(uri, takeFlags);

      // Validate file can be accessed via ContentResolver
      try (java.io.InputStream inputStream = getContentResolver().openInputStream(uri))
      {
        if (inputStream == null)
        {
          Toast.makeText(this, "Cannot access file", Toast.LENGTH_LONG).show();
          return;
        }

        // Read a few bytes to verify it's accessible
        byte[] header = new byte[4];
        int bytesRead = inputStream.read(header);
        if (bytesRead < 4)
        {
          Toast.makeText(this, "File is empty or unreadable", Toast.LENGTH_LONG).show();
          return;
        }
      }

      // Get filename from URI
      String filename = null;
      try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null))
      {
        if (cursor != null && cursor.moveToFirst())
        {
          int nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
          if (nameIndex >= 0)
          {
            filename = cursor.getString(nameIndex);
          }
        }
      }

      if (filename == null)
      {
        filename = uri.getLastPathSegment();
      }

      // Validate it's an ONNX file
      if (filename != null && !filename.toLowerCase().endsWith(".onnx"))
      {
        Toast.makeText(this, "File must be an .onnx file", Toast.LENGTH_SHORT).show();
        return;
      }

      // Save content URI to preferences (not file path!)
      SharedPreferences.Editor editor = getPreferenceManager().getSharedPreferences().edit();
      if (isEncoder)
      {
        editor.putString("neural_custom_encoder_uri", uri.toString());
        Toast.makeText(this, "✅ Encoder loaded: " + filename, Toast.LENGTH_SHORT).show();
      }
      else
      {
        editor.putString("neural_custom_decoder_uri", uri.toString());
        Toast.makeText(this, "✅ Decoder loaded: " + filename, Toast.LENGTH_SHORT).show();
      }
      editor.apply();

      // Update model info
      updateNeuralModelInfo();
    }
    catch (SecurityException e)
    {
      Toast.makeText(this, "❌ Permission denied. Please grant access to the file.", Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Security exception loading model", e);
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Error loading model: " + e.getMessage(), Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Error loading neural model file", e);
    }
  }

  /**
   * Perform backup to the selected URI
   */
  private void performBackup(Uri uri)
  {
    try
    {
      SharedPreferences prefs = getPreferenceManager().getSharedPreferences();
      boolean success = backupRestoreManager.exportConfig(uri, prefs);

      if (success)
      {
        int count = prefs.getAll().size();
        Toast.makeText(this, "Successfully exported " + count + " settings",
                       Toast.LENGTH_LONG).show();
      }
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Backup failed: " + e.getMessage(),
                     Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Backup failed", e);
    }
  }

  /**
   * Perform restore from the selected URI
   */
  private void performRestore(Uri uri)
  {
    try
    {
      SharedPreferences prefs = getPreferenceManager().getSharedPreferences();
      BackupRestoreManager.ImportResult result =
        backupRestoreManager.importConfig(uri, prefs);

      // Show results and warnings
      showRestoreResults(result);
    }
    catch (Exception e)
    {
      Toast.makeText(this, "Restore failed: " + e.getMessage(),
                     Toast.LENGTH_LONG).show();
      Log.e("SettingsActivity", "Restore failed", e);
    }
  }

  /**
   * Show restore results with warnings and restart prompt
   */
  private void showRestoreResults(BackupRestoreManager.ImportResult result)
  {
    StringBuilder message = new StringBuilder();
    message.append("Successfully restored ").append(result.importedCount).append(" settings");

    if (result.skippedCount > 0)
    {
      message.append("\n\nSkipped ").append(result.skippedCount)
             .append(" invalid or unrecognized settings");
    }

    // Warn about complex preferences that weren't restored
    boolean skippedLayouts = result.skippedKeys.contains("layouts");
    boolean skippedExtraKeys = result.skippedKeys.contains("extra_keys") ||
                               result.skippedKeys.contains("custom_extra_keys");

    if (skippedLayouts || skippedExtraKeys)
    {
      message.append("\n\n⚠️ Not restored:");
      if (skippedLayouts)
      {
        message.append("\n  • Keyboard layouts (needs manual reconfiguration)");
      }
      if (skippedExtraKeys)
      {
        message.append("\n  • Custom extra keys (needs manual reconfiguration)");
      }
    }

    if (!result.sourceVersion.equals("unknown"))
    {
      message.append("\n\nSource version: ").append(result.sourceVersion);
    }

    if (result.hasScreenSizeMismatch())
    {
      message.append("\n\n⚠️ Warning: Backup was from a device with different screen size. ")
             .append("Layout settings may need adjustment.");
    }

    message.append("\n\nThe app needs to restart to apply all settings correctly.");

    // Create dialog with restart option
    android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(this);
    builder.setTitle("Restore Complete");
    builder.setMessage(message.toString());
    builder.setCancelable(false);

    builder.setPositiveButton("Restart Now", (dialog, which) -> {
      // Restart the app
      Intent intent = getBaseContext().getPackageManager()
        .getLaunchIntentForPackage(getBaseContext().getPackageName());
      if (intent != null)
      {
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
      }
      finish();
      System.exit(0);
    });

    builder.setNegativeButton("Later", (dialog, which) -> {
      Toast.makeText(this, "Restart the app to apply all settings",
                     Toast.LENGTH_LONG).show();
    });

    builder.show();
  }

  /**
   * Setup CGR reset button functionality
   */
  private void setupCGRResetButtons()
  {
    // Reset to Optimal Values button
    Preference resetOptimalPref = findPreference("swipe_reset_optimal");
    if (resetOptimalPref != null)
    {
      resetOptimalPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          resetToSwipeOptimal();
          return true;
        }
      });
    }

    // Reset to Strict Values button
    Preference resetStrictPref = findPreference("swipe_reset_strict");
    if (resetStrictPref != null)
    {
      resetStrictPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          resetToSwipeStrict();
          return true;
        }
      });
    }

    // v1.33.8: Reset swipe corrections to defaults button
    Preference resetCorrectionsRef = findPreference("reset_swipe_corrections");
    if (resetCorrectionsRef != null)
    {
      resetCorrectionsRef.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener()
      {
        @Override
        public boolean onPreferenceClick(Preference preference)
        {
          resetSwipeCorrections();
          return true;
        }
      });
    }
  }
  
  /**
   * Reset to balanced keyboard swipe recognition values
   */
  private void resetToSwipeOptimal()
  {
    SharedPreferences prefs = getPreferenceManager().getSharedPreferences();
    SharedPreferences.Editor editor = prefs.edit();
    
    // Set balanced values for keyboard gesture recognition
    editor.putInt("proximity_weight", 100);      // Balanced proximity requirement
    editor.putInt("missing_key_penalty", 1000);  // Strong penalty for missing letters (10.0)
    editor.putInt("extra_key_penalty", 200);     // Moderate penalty for extra letters (2.0)
    editor.putInt("order_penalty", 500);         // Moderate order enforcement (5.0)
    editor.putInt("start_point_weight", 300);    // Strong start emphasis (3.0)
    editor.putInt("key_zone_radius", 120);       // Large detection area
    editor.putInt("path_sample_distance", 10);   // Frequent sampling
    editor.apply();
    
    // Force UI refresh
    recreate();
    
    Toast.makeText(this, "Reset to optimal keyboard recognition values", Toast.LENGTH_SHORT).show();
  }
  
  /**
   * Reset to strict keyboard swipe recognition values
   */
  private void resetToSwipeStrict()
  {
    SharedPreferences prefs = getPreferenceManager().getSharedPreferences();
    SharedPreferences.Editor editor = prefs.edit();

    // Set strict values for precise recognition
    editor.putInt("proximity_weight", 200);      // High proximity requirement
    editor.putInt("missing_key_penalty", 1500);  // Very strong penalty for missing letters (15.0)
    editor.putInt("extra_key_penalty", 500);     // Higher penalty for extra letters (5.0)
    editor.putInt("order_penalty", 1000);        // Strict order enforcement (10.0)
    editor.putInt("start_point_weight", 500);    // Very strong start emphasis (5.0)
    editor.putInt("key_zone_radius", 80);        // Smaller detection area (precise)
    editor.putInt("path_sample_distance", 5);    // Very frequent sampling
    editor.apply();

    // Force UI refresh
    recreate();

    Toast.makeText(this, "Reset to strict recognition values", Toast.LENGTH_SHORT).show();
  }

  /**
   * Reset swipe correction settings to defaults
   * v1.33.8: Implements reset button for swipe corrections
   */
  private void resetSwipeCorrections()
  {
    SharedPreferences prefs = getPreferenceManager().getSharedPreferences();
    SharedPreferences.Editor editor = prefs.edit();

    // Reset to balanced preset
    editor.putString("swipe_correction_preset", "balanced");

    // Reset fuzzy matching parameters to defaults
    editor.putInt("autocorrect_max_length_diff", 2);
    editor.putInt("autocorrect_prefix_length", 2);
    editor.putInt("autocorrect_max_beam_candidates", 3);
    editor.putFloat("autocorrect_char_match_threshold", 0.67f);
    editor.putInt("autocorrect_confidence_min_frequency", 500);

    // Reset autocorrect toggles to defaults
    editor.putBoolean("swipe_beam_autocorrect_enabled", true);
    editor.putBoolean("swipe_final_autocorrect_enabled", true);
    editor.putString("swipe_fuzzy_match_mode", "edit_distance");

    // Reset scoring weights to defaults
    editor.putInt("swipe_prediction_source", 60); // Balanced (60% NN, 40% freq)
    editor.putFloat("swipe_common_words_boost", 1.3f);
    editor.putFloat("swipe_top5000_boost", 1.0f);
    editor.putFloat("swipe_rare_words_penalty", 0.75f);

    editor.apply();

    // Force UI refresh to show updated values
    recreate();

    Toast.makeText(this, "Reset all swipe correction settings to defaults", Toast.LENGTH_LONG).show();
    android.util.Log.d("SettingsActivity", "Reset swipe corrections to default values");
  }
  
  @Override
  public void onSharedPreferenceChanged(SharedPreferences prefs, String key)
  {
    // v1.33.8: Handle swipe correction preset changes
    if ("swipe_correction_preset".equals(key))
    {
      String preset = prefs.getString(key, "balanced");
      applySwipeCorrectionPreset(prefs, preset);
      Toast.makeText(this, "Applied \"" + preset + "\" correction preset", Toast.LENGTH_SHORT).show();
      return;
    }

    // Trigger CGR parameter reload when settings change
    if (key != null && key.startsWith("cgr_"))
    {
      android.util.Log.d("SettingsActivity", "CGR parameter changed: " + key);

      // Parameter changes will take effect on next keyboard restart
      android.util.Log.d("SettingsActivity", "CGR parameter " + key + " changed, will take effect on restart");

      // Show current value for verification and update summaries
      int currentValue = prefs.getInt(key, -1);
      Toast.makeText(this, "Updated " + key + " = " + currentValue + " (takes effect immediately)", Toast.LENGTH_LONG).show();

      // Update parameter summaries to show new values
      updateCGRParameterSummaries();
    }

    // No super call needed for interface method
  }

  /**
   * Apply swipe correction preset values to fuzzy matching parameters
   * v1.33.8: Implements the swipe_correction_preset functionality
   *
   * @param prefs SharedPreferences to update
   * @param preset "strict", "balanced", or "lenient"
   */
  private void applySwipeCorrectionPreset(SharedPreferences prefs, String preset)
  {
    SharedPreferences.Editor editor = prefs.edit();

    switch (preset)
    {
      case "strict":
        // Strict (High Accuracy) - Minimize false corrections
        editor.putInt("autocorrect_max_length_diff", 1);       // Very strict on length
        editor.putInt("autocorrect_prefix_length", 3);         // First 3 letters must match
        editor.putInt("autocorrect_max_beam_candidates", 2);   // Only check top 2
        editor.putFloat("autocorrect_char_match_threshold", 0.80f); // 80% chars must match
        android.util.Log.d("SettingsActivity", "Applied STRICT preset: length_diff=1, prefix=3, candidates=2, threshold=0.80");
        break;

      case "lenient":
        // Lenient (Flexible) - Maximize corrections, accept more false positives
        editor.putInt("autocorrect_max_length_diff", 4);       // Very forgiving
        editor.putInt("autocorrect_prefix_length", 1);         // Only first letter
        editor.putInt("autocorrect_max_beam_candidates", 5);   // Check more
        editor.putFloat("autocorrect_char_match_threshold", 0.55f); // Only 55% match needed
        android.util.Log.d("SettingsActivity", "Applied LENIENT preset: length_diff=4, prefix=1, candidates=5, threshold=0.55");
        break;

      case "balanced":
      default:
        // Balanced (Default) - Middle ground
        editor.putInt("autocorrect_max_length_diff", 2);       // Default
        editor.putInt("autocorrect_prefix_length", 2);         // Default
        editor.putInt("autocorrect_max_beam_candidates", 3);   // Default
        editor.putFloat("autocorrect_char_match_threshold", 0.67f); // Default
        android.util.Log.d("SettingsActivity", "Applied BALANCED preset: length_diff=2, prefix=2, candidates=3, threshold=0.67");
        break;
    }

    editor.apply();
  }
  
  @Override
  protected void onResume()
  {
    super.onResume();
    // Register for preference changes and update summaries
    getPreferenceManager().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
    updateCGRParameterSummaries();
  }
  
  @Override
  protected void onPause()
  {
    super.onPause();
    // Unregister to prevent memory leaks
    getPreferenceManager().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
  }
  
  /**
   * Update CGR parameter summaries with current values
   */
  private void updateCGRParameterSummaries()
  {
    SharedPreferences prefs = getPreferenceManager().getSharedPreferences();
    
    // Update E_SIGMA summary
    Preference eSigmaPref = findPreference("cgr_e_sigma_config");
    if (eSigmaPref != null)
    {
      int eSigma = prefs.getInt("cgr_e_sigma", 120);
      eSigmaPref.setSummary("Current: " + eSigma + " (Position tolerance)");
    }
    
    // Update BETA summary  
    Preference betaPref = findPreference("cgr_beta_config");
    if (betaPref != null)
    {
      int beta = prefs.getInt("cgr_beta", 400);
      betaPref.setSummary("Current: " + beta + " (Variance ratio)");
    }
    
    // Update LAMBDA summary
    Preference lambdaPref = findPreference("cgr_lambda_config");
    if (lambdaPref != null)
    {
      int lambda = prefs.getInt("cgr_lambda", 65);
      lambdaPref.setSummary("Current: " + lambda + "% (Distance balance)");
    }
    
    // Update KAPPA summary
    Preference kappaPref = findPreference("cgr_kappa_config");
    if (kappaPref != null)
    {
      int kappa = prefs.getInt("cgr_kappa", 25);
      kappaPref.setSummary("Current: " + (kappa/10.0) + " (End-point bias)");
    }
    
    // Update Length Filter summary
    Preference lengthPref = findPreference("cgr_length_config");
    if (lengthPref != null)
    {
      int lengthFilter = prefs.getInt("cgr_length_filter", 70);
      lengthPref.setSummary("Current: " + lengthFilter + "% (Length similarity filter)");
    }
  }
}
