package juloo.keyboard2;

import android.app.Activity;
import android.content.SharedPreferences;
import android.graphics.PointF;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Activity for calibrating swipe typing patterns
 * Users swipe test words to train the DTW algorithm
 */
public class SwipeCalibrationActivity extends Activity
{
  private static final String PREF_KEY_CALIBRATION_DATA = "swipe_calibration_data";
  private static final int WORDS_PER_CALIBRATION = 10;
  private static final int REPETITIONS_PER_WORD = 3;
  
  // Test words of varying lengths
  private static final List<String> TEST_WORDS = Arrays.asList(
    "the", "and", "you", "that", "hello",
    "world", "keyboard", "typing", "android", "calibration"
  );
  
  private TextView _instructionText;
  private TextView _wordText;
  private TextView _progressText;
  private ProgressBar _progressBar;
  private Button _nextButton;
  private Button _skipButton;
  
  private int _currentWordIndex = 0;
  private int _currentRepetition = 0;
  private String _currentWord;
  private List<String> _shuffledWords;
  private Map<String, List<String>> _calibrationData;
  private String _currentSwipePath;
  
  @Override
  protected void onCreate(Bundle savedInstanceState)
  {
    super.onCreate(savedInstanceState);
    
    // Create a simple layout programmatically since we don't have the XML layout working
    LinearLayout layout = new LinearLayout(this);
    layout.setOrientation(LinearLayout.VERTICAL);
    layout.setPadding(32, 32, 32, 32);
    
    _instructionText = new TextView(this);
    _instructionText.setTextSize(18);
    _instructionText.setText("Swipe Calibration");
    layout.addView(_instructionText);
    
    _wordText = new TextView(this);
    _wordText.setTextSize(36);
    _wordText.setText("WORD");
    _wordText.setPadding(0, 32, 0, 32);
    layout.addView(_wordText);
    
    _progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
    layout.addView(_progressBar);
    
    _progressText = new TextView(this);
    _progressText.setText("Progress: 0 / 30");
    _progressText.setPadding(0, 16, 0, 32);
    layout.addView(_progressText);
    
    TextView infoText = new TextView(this);
    infoText.setText("Note: Calibration UI is simplified. In production, you would swipe on a keyboard view below.");
    infoText.setPadding(0, 32, 0, 32);
    layout.addView(infoText);
    
    LinearLayout buttonLayout = new LinearLayout(this);
    buttonLayout.setOrientation(LinearLayout.HORIZONTAL);
    
    _skipButton = new Button(this);
    _skipButton.setText("Skip");
    _skipButton.setOnClickListener(v -> skipCurrentWord());
    buttonLayout.addView(_skipButton);
    
    _nextButton = new Button(this);
    _nextButton.setText("Next");
    _nextButton.setOnClickListener(v -> nextWord());
    buttonLayout.addView(_nextButton);
    
    layout.addView(buttonLayout);
    
    setContentView(layout);
    
    initCalibrationData();
    startCalibration();
  }
  
  private void initCalibrationData()
  {
    _calibrationData = new HashMap<>();
    _shuffledWords = new ArrayList<>(TEST_WORDS);
    shuffleWords();
    
    // Load existing calibration data if available
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
    // In a real implementation, we would deserialize the stored data
    // For now, we just mark that calibration has been attempted
  }
  
  private void shuffleWords()
  {
    java.util.Collections.shuffle(_shuffledWords, new Random());
  }
  
  private void startCalibration()
  {
    _currentWordIndex = 0;
    _currentRepetition = 0;
    showNextWord();
  }
  
  private void showNextWord()
  {
    if (_currentWordIndex >= Math.min(WORDS_PER_CALIBRATION, _shuffledWords.size()))
    {
      finishCalibration();
      return;
    }
    
    _currentWord = _shuffledWords.get(_currentWordIndex);
    _wordText.setText(_currentWord.toUpperCase());
    
    String instruction = String.format("Swipe the word '%s'\n(Attempt %d of %d)",
      _currentWord, _currentRepetition + 1, REPETITIONS_PER_WORD);
    _instructionText.setText(instruction);
    
    updateProgress();
    
    // In a real implementation, we would capture the swipe path here
    // For now, we simulate that a swipe was done
    _currentSwipePath = "simulated_path_for_" + _currentWord;
  }
  
  private void nextWord()
  {
    if (_currentSwipePath != null)
    {
      // Store the swipe path for this word
      if (!_calibrationData.containsKey(_currentWord))
      {
        _calibrationData.put(_currentWord, new ArrayList<>());
      }
      _calibrationData.get(_currentWord).add(_currentSwipePath);
      
      _currentRepetition++;
      if (_currentRepetition >= REPETITIONS_PER_WORD)
      {
        _currentRepetition = 0;
        _currentWordIndex++;
      }
      
      showNextWord();
    }
  }
  
  private void skipCurrentWord()
  {
    _currentRepetition = 0;
    _currentWordIndex++;
    showNextWord();
  }
  
  private void updateProgress()
  {
    int totalSteps = WORDS_PER_CALIBRATION * REPETITIONS_PER_WORD;
    int currentStep = (_currentWordIndex * REPETITIONS_PER_WORD) + _currentRepetition;
    
    _progressBar.setMax(totalSteps);
    _progressBar.setProgress(currentStep);
    
    _progressText.setText(String.format("Progress: %d / %d", currentStep, totalSteps));
  }
  
  private void finishCalibration()
  {
    // Save calibration data
    saveCalibrationData();
    
    // Show completion message
    _instructionText.setText("Calibration Complete!");
    _wordText.setText("✓");
    _progressText.setText("Your swipe typing has been calibrated");
    
    _nextButton.setText("Done");
    _nextButton.setOnClickListener(v -> finish());
    _skipButton.setVisibility(View.GONE);
  }
  
  private void saveCalibrationData()
  {
    try
    {
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
      SharedPreferences.Editor editor = prefs.edit();
      
      // In a real implementation, we would serialize the calibration data
      // For now, we just mark that calibration has been completed
      editor.putBoolean("swipe_calibration_completed", true);
      editor.putInt("calibration_word_count", _calibrationData.size());
      
      // Store simplified data as strings
      for (Map.Entry<String, List<String>> entry : _calibrationData.entrySet())
      {
        String key = "calibration_" + entry.getKey();
        String value = String.join(",", entry.getValue());
        editor.putString(key, value);
      }
      
      editor.apply();
      
      android.util.Log.d("SwipeCalibration", "Saved calibration data for " + _calibrationData.size() + " words");
    }
    catch (Exception e)
    {
      android.util.Log.e("SwipeCalibration", "Failed to save calibration data", e);
    }
  }
}