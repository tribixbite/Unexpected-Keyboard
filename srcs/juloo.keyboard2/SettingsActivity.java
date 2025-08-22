package juloo.keyboard2;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.os.Build;
import android.os.Bundle;
import android.preference.Preference;
import android.preference.PreferenceActivity;
import android.preference.PreferenceManager;
import android.widget.Toast;
import java.io.File;
import juloo.keyboard2.ml.SwipeMLDataStore;
import juloo.keyboard2.ml.SwipeMLTrainer;
import android.app.ProgressDialog;

public class SettingsActivity extends PreferenceActivity
{
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

    boolean foldableDevice = FoldStateTracker.isFoldableDevice(this);
    findPreference("margin_bottom_portrait_unfolded").setEnabled(foldableDevice);
    findPreference("margin_bottom_landscape_unfolded").setEnabled(foldableDevice);
    findPreference("horizontal_margin_portrait_unfolded").setEnabled(foldableDevice);
    findPreference("horizontal_margin_landscape_unfolded").setEnabled(foldableDevice);
    findPreference("keyboard_height_unfolded").setEnabled(foldableDevice);
    findPreference("keyboard_height_landscape_unfolded").setEnabled(foldableDevice);
    
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
}
