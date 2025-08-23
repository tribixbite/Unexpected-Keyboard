package juloo.keyboard2;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PathMeasure;
import android.graphics.PointF;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import juloo.keyboard2.ml.SwipeMLData;
import juloo.keyboard2.ml.SwipeMLDataStore;

/**
 * Complete swipe calibration implementation with keyboard visualization
 */
public class SwipeCalibrationActivity extends Activity
{
  private static final String TAG = "SwipeCalibration";
  private static final String PREF_KEY_CALIBRATION_DATA = "swipe_calibration_data";
  private static final String LOG_FILE = "/data/data/com.termux/files/home/calibration_log.txt";
  
  // Calibration settings
  private static final int WORDS_PER_SESSION = 10;
  private static final int REPS_PER_WORD = 2;
  
  // Test words for calibration - will be replaced by frequency-based selection
  private static final List<String> FALLBACK_WORDS = Arrays.asList(
    "the", "and", "you", "that", "this",
    "hello", "world", "thanks", "keyboard", "android",
    "swipe", "typing", "calibration", "test", "quick"
  );
  
  // Top frequent words from dictionary (loaded dynamically)
  private List<String> _frequentWords = new ArrayList<>();
  private Map<String, Integer> _wordFrequencies = new HashMap<>();
  
  // QWERTY layout with actual key positions
  private static final String[][] KEYBOARD_LAYOUT = {
    {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"},
    {"a", "s", "d", "f", "g", "h", "j", "k", "l"},
    {"z", "x", "c", "v", "b", "n", "m"}
  };
  
  // UI Components
  private TextView _instructionText;
  private TextView _currentWordText;
  private TextView _progressText;
  private TextView _scoreText;
  private ProgressBar _progressBar;
  private KeyboardView _keyboardView;
  private Button _nextButton;
  private Button _skipButton;
  private Button _saveButton;
  private Button _deleteButton;
  private LinearLayout _scoreLayout;
  
  // Real-time accuracy metrics components
  private LinearLayout _metricsLayout;
  private TextView _sessionAccuracyText;
  private TextView _overallAccuracyText;
  private TextView _wpmText;
  private TextView _confusionPatternsText;
  private ProgressBar _accuracyProgressBar;
  
  // Calibration state
  private int _currentIndex = 0;
  private int _currentRep = 0;
  private String _currentWord;
  private List<String> _sessionWords;
  private Map<String, List<SwipePattern>> _calibrationData;
  private BufferedWriter _logWriter;
  private List<PointF> _currentSwipePoints;
  private long _swipeStartTime;
  private SwipeMLDataStore _mlDataStore;
  private int _screenWidth;
  private int _screenHeight;
  private int _keyboardHeight;
  private DTWPredictor _dtwPredictor;
  
  // Accuracy metrics tracking
  private int _sessionCorrectCount = 0;
  private int _sessionTotalCount = 0;
  private int _overallCorrectCount = 0;
  private int _overallTotalCount = 0;
  private List<Long> _swipeDurations = new ArrayList<>();
  private Map<String, Integer> _confusionMatrix = new HashMap<>();
  private long _sessionStartTime;
  private Handler _uiHandler;
  private Path _displayedSwipePath;
  private boolean _showingSwipeOverlay;
  private float _characterSize;
  private float _labelTextSize;
  private float _keyVerticalMargin;
  private float _keyHorizontalMargin;
  
  // Algorithm weight controls
  private LinearLayout _weightsLayout;
  private android.widget.SeekBar _dtwWeightSlider;
  private android.widget.SeekBar _gaussianWeightSlider;
  private android.widget.SeekBar _ngramWeightSlider;
  private android.widget.SeekBar _frequencyWeightSlider;
  private TextView _dtwWeightText;
  private TextView _gaussianWeightText;
  private TextView _ngramWeightText;
  private TextView _frequencyWeightText;
  private float _dtwWeight = 0.4f;
  private float _gaussianWeight = 0.3f;
  private float _ngramWeight = 0.2f;
  private float _frequencyWeight = 0.1f;
  
  @Override
  protected void onCreate(Bundle savedInstanceState)
  {
    super.onCreate(savedInstanceState);
    
    Log.d(TAG, "=== CALIBRATION ACTIVITY STARTED ===");
    initializeLogging();
    
    // Initialize ML data store
    _mlDataStore = SwipeMLDataStore.getInstance(this);
    
    // Initialize DTW predictor for scoring
    _dtwPredictor = new DTWPredictor(null);
    SwipeWeightConfig weightConfig = SwipeWeightConfig.getInstance(this);
    _dtwPredictor.setWeightConfig(weightConfig);
    
    // Initialize UI handler for delayed operations
    _uiHandler = new Handler();
    
    // Load frequency dictionary
    loadFrequencyDictionary();
    
    // Get screen dimensions
    android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
    getWindowManager().getDefaultDisplay().getMetrics(metrics);
    _screenWidth = metrics.widthPixels;
    _screenHeight = metrics.heightPixels;
    
    // Load user's keyboard configuration from preferences
    SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
    
    // Get keyboard height percentage from user settings
    int keyboardHeightPref = prefs.getInt("keyboard_height", 35);
    // Check for landscape mode
    boolean isLandscape = getResources().getConfiguration().orientation == 
                          android.content.res.Configuration.ORIENTATION_LANDSCAPE;
    if (isLandscape)
    {
      keyboardHeightPref = prefs.getInt("keyboard_height_landscape", 50);
    }
    
    // Calculate keyboard height
    float keyboardHeightPercent = keyboardHeightPref / 100.0f;
    _keyboardHeight = (int)(_screenHeight * keyboardHeightPercent);
    
    // Load user's text and margin settings
    try {
      _characterSize = Float.valueOf(prefs.getString("character_size", "1.15"));
    } catch (NumberFormatException e) {
      _characterSize = 1.15f;
    }
    _labelTextSize = 0.33f; // Default label text size
    
    try {
      _keyVerticalMargin = Float.valueOf(prefs.getString("key_vertical_margin", "1.5")) / 100;
    } catch (NumberFormatException e) {
      _keyVerticalMargin = 0.015f;
    }
    
    try {
      _keyHorizontalMargin = Float.valueOf(prefs.getString("key_horizontal_margin", "2")) / 100;
    } catch (NumberFormatException e) {
      _keyHorizontalMargin = 0.02f;
    }
    
    // Create main layout
    android.widget.RelativeLayout mainLayout = new android.widget.RelativeLayout(this);
    mainLayout.setBackgroundColor(Color.BLACK);
    
    // Create top content layout
    LinearLayout topLayout = new LinearLayout(this);
    topLayout.setOrientation(LinearLayout.VERTICAL);
    topLayout.setPadding(40, 40, 40, 20);
    android.widget.RelativeLayout.LayoutParams topParams = new android.widget.RelativeLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    topParams.addRule(android.widget.RelativeLayout.ALIGN_PARENT_TOP);
    topLayout.setLayoutParams(topParams);
    
    // Title
    TextView title = new TextView(this);
    title.setText("Swipe Calibration");
    title.setTextSize(24);
    title.setTextColor(Color.WHITE);
    title.setPadding(0, 0, 0, 20);
    topLayout.addView(title);
    
    // Instructions
    _instructionText = new TextView(this);
    _instructionText.setText("Swipe the word shown below - auto-advances on completion");
    _instructionText.setTextColor(Color.GRAY);
    _instructionText.setPadding(0, 0, 0, 10);
    topLayout.addView(_instructionText);
    
    // Current word display
    _currentWordText = new TextView(this);
    _currentWordText.setTextSize(32);
    _currentWordText.setTextColor(Color.CYAN);
    _currentWordText.setPadding(0, 20, 0, 20);
    topLayout.addView(_currentWordText);
    
    // Progress
    _progressText = new TextView(this);
    _progressText.setTextColor(Color.WHITE);
    topLayout.addView(_progressText);
    
    _progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
    _progressBar.setMax(WORDS_PER_SESSION * REPS_PER_WORD);
    topLayout.addView(_progressBar);
    
    // Score display layout
    _scoreLayout = new LinearLayout(this);
    _scoreLayout.setOrientation(LinearLayout.HORIZONTAL);
    _scoreLayout.setPadding(0, 10, 0, 10);
    _scoreLayout.setVisibility(View.GONE);
    
    TextView scoreLabel = new TextView(this);
    scoreLabel.setText("Prediction Score: ");
    scoreLabel.setTextColor(Color.WHITE);
    _scoreLayout.addView(scoreLabel);
    
    _scoreText = new TextView(this);
    _scoreText.setTextColor(Color.YELLOW);
    _scoreText.setTextSize(16);
    _scoreLayout.addView(_scoreText);
    
    topLayout.addView(_scoreLayout);
    
    // Real-time accuracy metrics display
    _metricsLayout = new LinearLayout(this);
    _metricsLayout.setOrientation(LinearLayout.VERTICAL);
    _metricsLayout.setPadding(0, 15, 0, 10);
    _metricsLayout.setBackgroundColor(0x33000000); // Semi-transparent background
    
    TextView metricsTitle = new TextView(this);
    metricsTitle.setText("📊 Real-Time Metrics");
    metricsTitle.setTextColor(Color.WHITE);
    metricsTitle.setTextSize(16);
    metricsTitle.setPadding(10, 5, 10, 5);
    _metricsLayout.addView(metricsTitle);
    
    // Session accuracy
    LinearLayout sessionAccuracyLayout = new LinearLayout(this);
    sessionAccuracyLayout.setOrientation(LinearLayout.HORIZONTAL);
    sessionAccuracyLayout.setPadding(10, 5, 10, 5);
    
    TextView sessionLabel = new TextView(this);
    sessionLabel.setText("Session Accuracy: ");
    sessionLabel.setTextColor(Color.GRAY);
    sessionAccuracyLayout.addView(sessionLabel);
    
    _sessionAccuracyText = new TextView(this);
    _sessionAccuracyText.setText("0%");
    _sessionAccuracyText.setTextColor(Color.CYAN);
    _sessionAccuracyText.setTextSize(18);
    sessionAccuracyLayout.addView(_sessionAccuracyText);
    
    _metricsLayout.addView(sessionAccuracyLayout);
    
    // Overall accuracy with progress bar
    LinearLayout overallAccuracyLayout = new LinearLayout(this);
    overallAccuracyLayout.setOrientation(LinearLayout.HORIZONTAL);
    overallAccuracyLayout.setPadding(10, 5, 10, 5);
    
    TextView overallLabel = new TextView(this);
    overallLabel.setText("Overall Accuracy: ");
    overallLabel.setTextColor(Color.GRAY);
    overallAccuracyLayout.addView(overallLabel);
    
    _overallAccuracyText = new TextView(this);
    _overallAccuracyText.setText("0%");
    _overallAccuracyText.setTextColor(Color.GREEN);
    _overallAccuracyText.setTextSize(18);
    overallAccuracyLayout.addView(_overallAccuracyText);
    
    _metricsLayout.addView(overallAccuracyLayout);
    
    _accuracyProgressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
    _accuracyProgressBar.setMax(100);
    _accuracyProgressBar.setProgress(0);
    LinearLayout.LayoutParams progParams = new LinearLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT, 10);
    progParams.setMargins(10, 0, 10, 5);
    _accuracyProgressBar.setLayoutParams(progParams);
    _metricsLayout.addView(_accuracyProgressBar);
    
    // Words per minute
    LinearLayout wpmLayout = new LinearLayout(this);
    wpmLayout.setOrientation(LinearLayout.HORIZONTAL);
    wpmLayout.setPadding(10, 5, 10, 5);
    
    TextView wpmLabel = new TextView(this);
    wpmLabel.setText("Speed (WPM): ");
    wpmLabel.setTextColor(Color.GRAY);
    wpmLayout.addView(wpmLabel);
    
    _wpmText = new TextView(this);
    _wpmText.setText("0");
    _wpmText.setTextColor(Color.YELLOW);
    _wpmText.setTextSize(16);
    wpmLayout.addView(_wpmText);
    
    _metricsLayout.addView(wpmLayout);
    
    // Confusion patterns
    _confusionPatternsText = new TextView(this);
    _confusionPatternsText.setText("Common errors will appear here");
    _confusionPatternsText.setTextColor(Color.LTGRAY);
    _confusionPatternsText.setTextSize(12);
    _confusionPatternsText.setPadding(10, 5, 10, 5);
    _metricsLayout.addView(_confusionPatternsText);
    
    topLayout.addView(_metricsLayout);
    
    // Algorithm weight controls
    _weightsLayout = new LinearLayout(this);
    _weightsLayout.setOrientation(LinearLayout.VERTICAL);
    _weightsLayout.setPadding(0, 10, 0, 10);
    
    TextView weightsTitle = new TextView(this);
    weightsTitle.setText("Algorithm Weights:");
    weightsTitle.setTextColor(Color.WHITE);
    weightsTitle.setTextSize(14);
    _weightsLayout.addView(weightsTitle);
    
    // DTW Weight
    LinearLayout dtwLayout = new LinearLayout(this);
    dtwLayout.setOrientation(LinearLayout.HORIZONTAL);
    _dtwWeightText = new TextView(this);
    _dtwWeightText.setText("DTW: 40%");
    _dtwWeightText.setTextColor(Color.GRAY);
    _dtwWeightText.setLayoutParams(new LinearLayout.LayoutParams(150, ViewGroup.LayoutParams.WRAP_CONTENT));
    dtwLayout.addView(_dtwWeightText);
    _dtwWeightSlider = new android.widget.SeekBar(this);
    _dtwWeightSlider.setMax(100);
    _dtwWeightSlider.setProgress(40);
    _dtwWeightSlider.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
    _dtwWeightSlider.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
        _dtwWeight = progress / 100.0f;
        _dtwWeightText.setText("DTW: " + progress + "%");
        normalizeWeights();
      }
      @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
      @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
    });
    dtwLayout.addView(_dtwWeightSlider);
    _weightsLayout.addView(dtwLayout);
    
    // Gaussian Weight
    LinearLayout gaussianLayout = new LinearLayout(this);
    gaussianLayout.setOrientation(LinearLayout.HORIZONTAL);
    _gaussianWeightText = new TextView(this);
    _gaussianWeightText.setText("Gaussian: 30%");
    _gaussianWeightText.setTextColor(Color.GRAY);
    _gaussianWeightText.setLayoutParams(new LinearLayout.LayoutParams(150, ViewGroup.LayoutParams.WRAP_CONTENT));
    gaussianLayout.addView(_gaussianWeightText);
    _gaussianWeightSlider = new android.widget.SeekBar(this);
    _gaussianWeightSlider.setMax(100);
    _gaussianWeightSlider.setProgress(30);
    _gaussianWeightSlider.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
    _gaussianWeightSlider.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
        _gaussianWeight = progress / 100.0f;
        _gaussianWeightText.setText("Gaussian: " + progress + "%");
        normalizeWeights();
      }
      @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
      @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
    });
    gaussianLayout.addView(_gaussianWeightSlider);
    _weightsLayout.addView(gaussianLayout);
    
    // N-gram Weight
    LinearLayout ngramLayout = new LinearLayout(this);
    ngramLayout.setOrientation(LinearLayout.HORIZONTAL);
    _ngramWeightText = new TextView(this);
    _ngramWeightText.setText("N-gram: 20%");
    _ngramWeightText.setTextColor(Color.GRAY);
    _ngramWeightText.setLayoutParams(new LinearLayout.LayoutParams(150, ViewGroup.LayoutParams.WRAP_CONTENT));
    ngramLayout.addView(_ngramWeightText);
    _ngramWeightSlider = new android.widget.SeekBar(this);
    _ngramWeightSlider.setMax(100);
    _ngramWeightSlider.setProgress(20);
    _ngramWeightSlider.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
    _ngramWeightSlider.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
        _ngramWeight = progress / 100.0f;
        _ngramWeightText.setText("N-gram: " + progress + "%");
        normalizeWeights();
      }
      @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
      @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
    });
    ngramLayout.addView(_ngramWeightSlider);
    _weightsLayout.addView(ngramLayout);
    
    // Frequency Weight
    LinearLayout freqLayout = new LinearLayout(this);
    freqLayout.setOrientation(LinearLayout.HORIZONTAL);
    _frequencyWeightText = new TextView(this);
    _frequencyWeightText.setText("Frequency: 10%");
    _frequencyWeightText.setTextColor(Color.GRAY);
    _frequencyWeightText.setLayoutParams(new LinearLayout.LayoutParams(150, ViewGroup.LayoutParams.WRAP_CONTENT));
    freqLayout.addView(_frequencyWeightText);
    _frequencyWeightSlider = new android.widget.SeekBar(this);
    _frequencyWeightSlider.setMax(100);
    _frequencyWeightSlider.setProgress(10);
    _frequencyWeightSlider.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f));
    _frequencyWeightSlider.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
        _frequencyWeight = progress / 100.0f;
        _frequencyWeightText.setText("Frequency: " + progress + "%");
        normalizeWeights();
      }
      @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
      @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
    });
    freqLayout.addView(_frequencyWeightSlider);
    _weightsLayout.addView(freqLayout);
    
    topLayout.addView(_weightsLayout);
    
    // Buttons above keyboard
    LinearLayout buttonLayout = new LinearLayout(this);
    buttonLayout.setOrientation(LinearLayout.HORIZONTAL);
    buttonLayout.setPadding(0, 20, 0, 0);
    
    _skipButton = new Button(this);
    _skipButton.setText("Skip Word");
    _skipButton.setOnClickListener(v -> skipWord());
    buttonLayout.addView(_skipButton);
    
    _nextButton = new Button(this);
    _nextButton.setText("Retry");
    _nextButton.setEnabled(true);
    _nextButton.setOnClickListener(v -> {
      _keyboardView.reset();
      Toast.makeText(this, "Try swiping again", Toast.LENGTH_SHORT).show();
    });
    buttonLayout.addView(_nextButton);
    
    _saveButton = new Button(this);
    _saveButton.setText("Save & Exit");
    _saveButton.setOnClickListener(v -> saveAndExit());
    buttonLayout.addView(_saveButton);
    
    _deleteButton = new Button(this);
    _deleteButton.setText("Delete Samples");
    _deleteButton.setOnClickListener(v -> confirmDeleteSamples());
    buttonLayout.addView(_deleteButton);
    
    topLayout.addView(buttonLayout);
    mainLayout.addView(topLayout);
    
    // Keyboard at bottom with real dimensions
    _keyboardView = new KeyboardView(this);
    android.widget.RelativeLayout.LayoutParams keyboardParams = new android.widget.RelativeLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT, _keyboardHeight);
    keyboardParams.addRule(android.widget.RelativeLayout.ALIGN_PARENT_BOTTOM);
    _keyboardView.setLayoutParams(keyboardParams);
    mainLayout.addView(_keyboardView);
    
    setContentView(mainLayout);
    
    // Initialize calibration
    initializeCalibration();
    
    // Load saved weights
    loadSavedWeights();
  }
  
  /**
   * Normalize weights to sum to 100%
   */
  private void normalizeWeights()
  {
    float total = _dtwWeight + _gaussianWeight + _ngramWeight + _frequencyWeight;
    if (total > 0)
    {
      // Update display text with normalized values
      int dtwPercent = Math.round(_dtwWeight / total * 100);
      int gaussianPercent = Math.round(_gaussianWeight / total * 100);
      int ngramPercent = Math.round(_ngramWeight / total * 100);
      int freqPercent = Math.round(_frequencyWeight / total * 100);
      
      // Ensure they sum to 100
      int sum = dtwPercent + gaussianPercent + ngramPercent + freqPercent;
      if (sum != 100)
      {
        dtwPercent += (100 - sum);
      }
      
      _dtwWeightText.setText("DTW: " + dtwPercent + "%");
      _gaussianWeightText.setText("Gaussian: " + gaussianPercent + "%");
      _ngramWeightText.setText("N-gram: " + ngramPercent + "%");
      _frequencyWeightText.setText("Frequency: " + freqPercent + "%");
    }
    
    // Save weights and update DTW predictor
    saveWeights();
    
    // Update DTW predictor with new weights
    SwipeWeightConfig weightConfig = SwipeWeightConfig.getInstance(this);
    weightConfig.saveWeights(_dtwWeight, _gaussianWeight, _ngramWeight, _frequencyWeight);
    _dtwPredictor.setWeightConfig(weightConfig);
  }
  
  /**
   * Save algorithm weights to preferences
   */
  private void saveWeights()
  {
    SharedPreferences prefs = getSharedPreferences("swipe_weights", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = prefs.edit();
    editor.putFloat("dtw_weight", _dtwWeight);
    editor.putFloat("gaussian_weight", _gaussianWeight);
    editor.putFloat("ngram_weight", _ngramWeight);
    editor.putFloat("frequency_weight", _frequencyWeight);
    editor.apply();
  }
  
  /**
   * Load saved algorithm weights
   */
  private void loadSavedWeights()
  {
    SharedPreferences prefs = getSharedPreferences("swipe_weights", Context.MODE_PRIVATE);
    _dtwWeight = prefs.getFloat("dtw_weight", 0.4f);
    _gaussianWeight = prefs.getFloat("gaussian_weight", 0.3f);
    _ngramWeight = prefs.getFloat("ngram_weight", 0.2f);
    _frequencyWeight = prefs.getFloat("frequency_weight", 0.1f);
    
    // Update sliders
    _dtwWeightSlider.setProgress(Math.round(_dtwWeight * 100));
    _gaussianWeightSlider.setProgress(Math.round(_gaussianWeight * 100));
    _ngramWeightSlider.setProgress(Math.round(_ngramWeight * 100));
    _frequencyWeightSlider.setProgress(Math.round(_frequencyWeight * 100));
    
    normalizeWeights();
  }
  
  private void initializeLogging()
  {
    try
    {
      _logWriter = new BufferedWriter(new FileWriter(LOG_FILE, true));
      _logWriter.write("\n=== CALIBRATION SESSION STARTED: " + new Date() + " ===\n");
      _logWriter.flush();
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to initialize logging: " + e.getMessage());
    }
  }
  
  private void initializeCalibration()
  {
    _calibrationData = new HashMap<>();
    _currentSwipePoints = new ArrayList<>();
    _sessionStartTime = System.currentTimeMillis();
    
    // Load overall accuracy from preferences
    loadOverallAccuracy();
    
    // Select words from top 30% most frequent
    if (_frequentWords.isEmpty())
    {
      // Fallback if dictionary didn't load
      _sessionWords = new ArrayList<>(FALLBACK_WORDS);
      Collections.shuffle(_sessionWords);
      _sessionWords = _sessionWords.subList(0, Math.min(WORDS_PER_SESSION, _sessionWords.size()));
    }
    else
    {
      // Use top 30% most frequent words
      int topCount = Math.max(50, _frequentWords.size() * 30 / 100);
      List<String> topWords = _frequentWords.subList(0, Math.min(topCount, _frequentWords.size()));
      
      // Filter for reasonable length (3-8 characters)
      List<String> candidates = new ArrayList<>();
      for (String word : topWords)
      {
        if (word.length() >= 3 && word.length() <= 8 && word.matches("[a-z]+"))
        {
          candidates.add(word);
        }
      }
      
      // Randomly select from candidates
      Collections.shuffle(candidates, new Random());
      _sessionWords = candidates.subList(0, Math.min(WORDS_PER_SESSION, candidates.size()));
    }
    
    logMessage("Session words: " + _sessionWords);
    
    // Start with first word
    _currentIndex = 0;
    _currentRep = 0;
    showNextWord();
  }
  
  private void showNextWord()
  {
    if (_currentIndex >= _sessionWords.size())
    {
      // Calibration complete
      showCompletionMessage();
      return;
    }
    
    _currentWord = _sessionWords.get(_currentIndex);
    _currentWordText.setText(_currentWord);
    
    int totalProgress = _currentIndex * REPS_PER_WORD + _currentRep;
    _progressBar.setProgress(totalProgress);
    _progressText.setText(String.format("Word %d of %d, Rep %d of %d",
      _currentIndex + 1, _sessionWords.size(),
      _currentRep + 1, REPS_PER_WORD));
    
    _nextButton.setEnabled(false);
    _keyboardView.reset();
    
    logMessage("Showing word: " + _currentWord + " (rep " + (_currentRep + 1) + ")");
  }
  
  private void recordSwipe(List<PointF> points)
  {
    if (points.isEmpty()) return;
    
    long duration = System.currentTimeMillis() - _swipeStartTime;
    
    // Show the swipe path overlay
    showSwipePathOverlay(points);
    
    // Create swipe pattern for legacy storage
    SwipePattern pattern = new SwipePattern(_currentWord, points, duration);
    
    // Add to calibration data
    if (!_calibrationData.containsKey(_currentWord))
    {
      _calibrationData.put(_currentWord, new ArrayList<>());
    }
    _calibrationData.get(_currentWord).add(pattern);
    
    // Create ML data object with actual keyboard dimensions
    // Get keyboard view position on screen for accurate coordinates
    int[] location = new int[2];
    _keyboardView.getLocationOnScreen(location);
    int keyboardY = location[1];
    
    SwipeMLData mlData = new SwipeMLData(_currentWord, "calibration",
                                         _screenWidth, _screenHeight, _keyboardHeight);
    
    // Add trace points with timestamps and normalized to keyboard space
    long startTime = _swipeStartTime;
    for (int i = 0; i < points.size(); i++)
    {
      PointF p = points.get(i);
      // Estimate timestamp based on linear interpolation
      long timestamp = startTime + (duration * i / Math.max(1, points.size() - 1));
      
      // Store both absolute coordinates and keyboard-relative coordinates
      mlData.addRawPoint(p.x, p.y + keyboardY, timestamp);
      
      // Add registered key
      String key = _keyboardView.getKeyAt(p.x, p.y);
      if (key != null)
      {
        mlData.addRegisteredKey(key);
      }
    }
    
    // Add keyboard dimensions metadata
    mlData.setKeyboardDimensions(_screenWidth, _keyboardHeight, keyboardY);
    
    // Store ML data
    _mlDataStore.storeSwipeData(mlData);
    
    // Log the swipe with ML info
    StringBuilder log = new StringBuilder();
    log.append("SWIPE RECORDED: word=").append(_currentWord);
    log.append(", points=").append(points.size());
    log.append(", duration=").append(duration).append("ms");
    log.append(", ML_trace_id=").append(mlData.getTraceId());
    log.append(", registered_keys=").append(mlData.getRegisteredKeys().size());
    log.append(", keyboard_height=").append(_keyboardHeight);
    log.append(", keyboard_y=").append(keyboardY);
    log.append(", path=[");
    
    for (String key : mlData.getRegisteredKeys())
    {
      log.append(key);
    }
    log.append("]");
    
    logMessage(log.toString());
    
    // Calculate prediction score
    calculateAndShowScore(points);
    
    // Show feedback
    Toast.makeText(this, "Swipe recorded! Duration: " + duration + "ms", Toast.LENGTH_SHORT).show();
    
    // Auto-advance after showing overlay (with delay)
    _uiHandler.postDelayed(() -> {
      _showingSwipeOverlay = false;
      _keyboardView.clearOverlay();
      nextWord();
    }, 1500); // Show overlay for 1.5 seconds
  }
  
  private void nextWord()
  {
    _currentRep++;
    if (_currentRep >= REPS_PER_WORD)
    {
      _currentRep = 0;
      _currentIndex++;
    }
    showNextWord();
  }
  
  private void skipWord()
  {
    logMessage("Skipped word: " + _currentWord);
    _currentRep = 0;
    _currentIndex++;
    showNextWord();
  }
  
  private void saveAndExit()
  {
    saveCalibrationData();
    finish();
  }
  
  private void saveCalibrationData()
  {
    SharedPreferences prefs = getSharedPreferences("swipe_calibration", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = prefs.edit();
    
    // Save calibration data
    StringBuilder data = new StringBuilder();
    for (Map.Entry<String, List<SwipePattern>> entry : _calibrationData.entrySet())
    {
      String word = entry.getKey();
      List<SwipePattern> patterns = entry.getValue();
      
      data.append(word).append(":");
      for (SwipePattern pattern : patterns)
      {
        data.append(pattern.serialize()).append(";");
      }
      data.append("\n");
    }
    
    editor.putString(PREF_KEY_CALIBRATION_DATA, data.toString());
    editor.putLong("calibration_timestamp", System.currentTimeMillis());
    editor.apply();
    
    logMessage("CALIBRATION SAVED: " + _calibrationData.size() + " words");
    logMessage("Data: " + data.toString());
    
    Toast.makeText(this, "Calibration data saved!", Toast.LENGTH_LONG).show();
  }
  
  private void showCompletionMessage()
  {
    _instructionText.setText("Calibration Complete!");
    _currentWordText.setText("✓");
    _currentWordText.setTextColor(Color.GREEN);
    _progressBar.setProgress(_progressBar.getMax());
    _nextButton.setEnabled(false);
    _skipButton.setEnabled(false);
    
    logMessage("=== CALIBRATION COMPLETE ===");
    logMessage("Total words calibrated: " + _calibrationData.size());
    
    for (Map.Entry<String, List<SwipePattern>> entry : _calibrationData.entrySet())
    {
      logMessage(entry.getKey() + ": " + entry.getValue().size() + " patterns");
    }
  }
  
  private void logMessage(String message)
  {
    Log.d(TAG, message);
    if (_logWriter != null)
    {
      try
      {
        _logWriter.write("[" + new Date() + "] " + message + "\n");
        _logWriter.flush();
      }
      catch (IOException e)
      {
        Log.e(TAG, "Failed to write log: " + e.getMessage());
      }
    }
  }
  
  /**
   * Load frequency dictionary for word selection
   */
  private void loadFrequencyDictionary()
  {
    try
    {
      BufferedReader reader = new BufferedReader(
        new InputStreamReader(getAssets().open("dictionaries/en_US_enhanced.txt")));
      
      List<WordFrequency> wordList = new ArrayList<>();
      String line;
      while ((line = reader.readLine()) != null)
      {
        String[] parts = line.trim().split("\t");
        if (parts.length >= 1)
        {
          String word = parts[0].toLowerCase();
          int frequency = parts.length > 1 ? Integer.parseInt(parts[1]) : 1000;
          wordList.add(new WordFrequency(word, frequency));
          _wordFrequencies.put(word, frequency);
        }
      }
      reader.close();
      
      // Sort by frequency (highest first)
      Collections.sort(wordList, (a, b) -> Integer.compare(b.frequency, a.frequency));
      
      // Extract just the words
      _frequentWords.clear();
      for (WordFrequency wf : wordList)
      {
        _frequentWords.add(wf.word);
      }
      
      // Load dictionary into DTW predictor
      _dtwPredictor.loadDictionary(_wordFrequencies);
      
      Log.d(TAG, "Loaded " + _frequentWords.size() + " words from dictionary");
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load dictionary: " + e.getMessage());
      _frequentWords.clear();
    }
  }
  
  /**
   * Show confirmation dialog for deleting stored samples
   */
  private void confirmDeleteSamples()
  {
    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    builder.setTitle("Delete Stored Samples");
    builder.setMessage("This will delete all stored calibration samples. Are you sure?");
    builder.setPositiveButton("Delete", new DialogInterface.OnClickListener() {
      @Override
      public void onClick(DialogInterface dialog, int which)
      {
        deleteStoredSamples();
      }
    });
    builder.setNegativeButton("Cancel", null);
    builder.show();
  }
  
  /**
   * Delete all stored calibration samples
   */
  private void deleteStoredSamples()
  {
    // Clear ML data store
    _mlDataStore.clearAllData();
    
    // Clear SharedPreferences
    SharedPreferences prefs = getSharedPreferences("swipe_calibration", Context.MODE_PRIVATE);
    prefs.edit().clear().apply();
    
    // Clear current session data
    _calibrationData.clear();
    
    Toast.makeText(this, "All calibration samples deleted", Toast.LENGTH_LONG).show();
    logMessage("All calibration samples deleted by user");
  }
  
  /**
   * Show the swipe path as an overlay on the keyboard
   */
  private void showSwipePathOverlay(List<PointF> points)
  {
    _showingSwipeOverlay = true;
    _displayedSwipePath = new Path();
    
    if (!points.isEmpty())
    {
      PointF first = points.get(0);
      _displayedSwipePath.moveTo(first.x, first.y);
      
      for (int i = 1; i < points.size(); i++)
      {
        PointF p = points.get(i);
        _displayedSwipePath.lineTo(p.x, p.y);
      }
    }
    
    _keyboardView.setSwipeOverlay(_displayedSwipePath);
  }
  
  /**
   * Calculate and display prediction score for the swipe
   */
  private void calculateAndShowScore(List<PointF> points)
  {
    if (_dtwPredictor == null || _currentWord == null)
      return;
    
    // Track swipe duration
    long swipeDuration = System.currentTimeMillis() - _swipeStartTime;
    _swipeDurations.add(swipeDuration);
    
    // Set keyboard dimensions for DTW predictor
    _dtwPredictor.setKeyboardDimensions(_keyboardView.getWidth(), _keyboardView.getHeight());
    
    // Get touched keys from keyboard view
    List<KeyboardData.Key> touchedKeys = new ArrayList<>();
    for (PointF p : points)
    {
      String keyChar = _keyboardView.getKeyAt(p.x, p.y);
      if (keyChar != null)
      {
        // Create a simple Key object for DTW predictor
        KeyValue kv = KeyValue.makeStringKey(keyChar);
        KeyboardData.Key key = new KeyboardData.Key(
          new KeyValue[]{kv, null, null, null, null},
          null, // anticircle
          0,    // flags
          1.0f, // width
          0.0f, // shift
          null  // indication
        );
        touchedKeys.add(key);
      }
    }
    
    // Get DTW predictions
    DTWPredictor.DTWResult result = _dtwPredictor.predictWithCoordinates(points, touchedKeys);
    
    // Find ranking of correct word
    int rank = -1;
    float score = 0.0f;
    String topPrediction = result.words.isEmpty() ? "" : result.words.get(0);
    
    for (int i = 0; i < result.words.size(); i++)
    {
      if (result.words.get(i).equals(_currentWord))
      {
        rank = i + 1; // 1-based ranking
        score = result.scores.get(i);
        break;
      }
    }
    
    // Update accuracy metrics
    _sessionTotalCount++;
    _overallTotalCount++;
    
    if (rank == 1)
    {
      _sessionCorrectCount++;
      _overallCorrectCount++;
    }
    else if (!topPrediction.isEmpty())
    {
      // Track confusion patterns
      String confusionKey = _currentWord + "→" + topPrediction;
      _confusionMatrix.put(confusionKey, _confusionMatrix.getOrDefault(confusionKey, 0) + 1);
    }
    
    // Update metrics display
    updateMetricsDisplay();
    
    // Display score
    _scoreLayout.setVisibility(View.VISIBLE);
    if (rank > 0)
    {
      String scoreText = String.format("Rank #%d (Score: %.2f, Confidence: %.0f%%)",
                                       rank, score, result.confidence * 100);
      _scoreText.setText(scoreText);
      
      // Color based on ranking
      if (rank == 1)
      {
        _scoreText.setTextColor(Color.GREEN);
      }
      else if (rank <= 3)
      {
        _scoreText.setTextColor(Color.YELLOW);
      }
      else
      {
        _scoreText.setTextColor(Color.RED);
      }
    }
    else
    {
      _scoreText.setText("Not in top 10 predictions");
      _scoreText.setTextColor(Color.RED);
    }
    
    // Log detailed results
    logMessage("Prediction results for '" + _currentWord + "': " +
               (rank > 0 ? "Rank #" + rank : "Not found") +
               ", Top 3: " + (result.words.size() >= 3 ?
                              result.words.subList(0, 3) : result.words));
    
    // Save overall accuracy periodically
    if (_overallTotalCount % 5 == 0)
    {
      saveOverallAccuracy();
    }
  }
  
  /**
   * Update the real-time metrics display
   */
  private void updateMetricsDisplay()
  {
    // Calculate session accuracy
    float sessionAccuracy = _sessionTotalCount > 0 ? 
      (float)_sessionCorrectCount / _sessionTotalCount * 100 : 0;
    _sessionAccuracyText.setText(String.format("%.1f%%", sessionAccuracy));
    
    // Color code session accuracy
    if (sessionAccuracy >= 80)
      _sessionAccuracyText.setTextColor(Color.GREEN);
    else if (sessionAccuracy >= 60)
      _sessionAccuracyText.setTextColor(Color.YELLOW);
    else
      _sessionAccuracyText.setTextColor(Color.RED);
    
    // Calculate overall accuracy
    float overallAccuracy = _overallTotalCount > 0 ?
      (float)_overallCorrectCount / _overallTotalCount * 100 : 0;
    _overallAccuracyText.setText(String.format("%.1f%%", overallAccuracy));
    _accuracyProgressBar.setProgress(Math.round(overallAccuracy));
    
    // Calculate WPM (words per minute)
    if (!_swipeDurations.isEmpty())
    {
      long avgDuration = 0;
      for (Long duration : _swipeDurations)
      {
        avgDuration += duration;
      }
      avgDuration /= _swipeDurations.size();
      
      // Calculate WPM (60000 ms per minute / avg duration per word)
      int wpm = avgDuration > 0 ? (int)(60000 / avgDuration) : 0;
      _wpmText.setText(String.valueOf(wpm));
      
      // Color code WPM
      if (wpm >= 40)
        _wpmText.setTextColor(Color.GREEN);
      else if (wpm >= 25)
        _wpmText.setTextColor(Color.YELLOW);
      else
        _wpmText.setTextColor(Color.WHITE);
    }
    
    // Update confusion patterns display
    updateConfusionPatterns();
  }
  
  /**
   * Update the confusion patterns display
   */
  private void updateConfusionPatterns()
  {
    if (_confusionMatrix.isEmpty())
    {
      _confusionPatternsText.setText("No errors yet - great job!");
      _confusionPatternsText.setTextColor(Color.GREEN);
      return;
    }
    
    // Find top 3 confusion patterns
    List<Map.Entry<String, Integer>> sortedConfusions = new ArrayList<>(_confusionMatrix.entrySet());
    Collections.sort(sortedConfusions, new Comparator<Map.Entry<String, Integer>>() {
      @Override
      public int compare(Map.Entry<String, Integer> a, Map.Entry<String, Integer> b) {
        return b.getValue().compareTo(a.getValue());
      }
    });
    
    StringBuilder patterns = new StringBuilder("Common errors: ");
    int count = 0;
    for (Map.Entry<String, Integer> entry : sortedConfusions)
    {
      if (count >= 3) break;
      if (count > 0) patterns.append(", ");
      patterns.append(entry.getKey()).append(" (").append(entry.getValue()).append("×)");
      count++;
    }
    
    _confusionPatternsText.setText(patterns.toString());
    _confusionPatternsText.setTextColor(Color.LTGRAY);
  }
  
  /**
   * Load overall accuracy from preferences
   */
  private void loadOverallAccuracy()
  {
    SharedPreferences prefs = getSharedPreferences("swipe_metrics", Context.MODE_PRIVATE);
    _overallCorrectCount = prefs.getInt("overall_correct", 0);
    _overallTotalCount = prefs.getInt("overall_total", 0);
    
    // Load confusion matrix
    String confusionData = prefs.getString("confusion_matrix", "");
    if (!confusionData.isEmpty())
    {
      String[] pairs = confusionData.split(";");
      for (String pair : pairs)
      {
        String[] parts = pair.split(":");
        if (parts.length == 2)
        {
          try
          {
            _confusionMatrix.put(parts[0], Integer.parseInt(parts[1]));
          }
          catch (NumberFormatException e)
          {
            // Ignore invalid entries
          }
        }
      }
    }
    
    // Update display with loaded data
    updateMetricsDisplay();
  }
  
  /**
   * Save overall accuracy to preferences
   */
  private void saveOverallAccuracy()
  {
    SharedPreferences prefs = getSharedPreferences("swipe_metrics", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = prefs.edit();
    editor.putInt("overall_correct", _overallCorrectCount);
    editor.putInt("overall_total", _overallTotalCount);
    
    // Save confusion matrix
    StringBuilder confusionData = new StringBuilder();
    for (Map.Entry<String, Integer> entry : _confusionMatrix.entrySet())
    {
      if (confusionData.length() > 0) confusionData.append(";");
      confusionData.append(entry.getKey()).append(":").append(entry.getValue());
    }
    editor.putString("confusion_matrix", confusionData.toString());
    
    editor.apply();
  }
  
  /**
   * Helper class for word frequency
   */
  private static class WordFrequency
  {
    final String word;
    final int frequency;
    
    WordFrequency(String word, int frequency)
    {
      this.word = word;
      this.frequency = frequency;
    }
  }
  
  @Override
  protected void onDestroy()
  {
    super.onDestroy();
    if (_logWriter != null)
    {
      try
      {
        _logWriter.write("=== CALIBRATION SESSION ENDED ===\n");
        _logWriter.close();
      }
      catch (IOException e)
      {
        Log.e(TAG, "Failed to close log writer: " + e.getMessage());
      }
    }
  }
  
  /**
   * Custom keyboard view for swipe input
   */
  private class KeyboardView extends View
  {
    private Paint _keyPaint;
    private Paint _keyBorderPaint;
    private Paint _textPaint;
    private Paint _swipePaint;
    private Paint _overlayPaint;
    private Map<String, KeyButton> _keys;
    private Path _swipePath;
    private Path _overlayPath;
    private List<PointF> _swipePoints;
    private boolean _swiping;
    
    public KeyboardView(Context context)
    {
      super(context);
      init();
    }
    
    private void init()
    {
      _keyPaint = new Paint();
      _keyPaint.setColor(0xFF2B2B2B); // Darker gray similar to real keyboard
      _keyPaint.setStyle(Paint.Style.FILL);
      _keyPaint.setAntiAlias(true);
      
      _keyBorderPaint = new Paint();
      _keyBorderPaint.setColor(0xFF1A1A1A); // Even darker for border
      _keyBorderPaint.setStyle(Paint.Style.STROKE);
      _keyBorderPaint.setStrokeWidth(2);
      _keyBorderPaint.setAntiAlias(true);
      
      _textPaint = new Paint();
      _textPaint.setColor(Color.WHITE);
      _textPaint.setTextAlign(Paint.Align.CENTER);
      _textPaint.setAntiAlias(true);
      _textPaint.setSubpixelText(true);
      // Text size will be set in onSizeChanged based on key dimensions
      
      _swipePaint = new Paint();
      _swipePaint.setColor(Color.CYAN);
      _swipePaint.setStrokeWidth(8);
      _swipePaint.setStyle(Paint.Style.STROKE);
      _swipePaint.setAlpha(180);
      _swipePaint.setAntiAlias(true);
      _swipePaint.setStrokeCap(Paint.Cap.ROUND);
      _swipePaint.setStrokeJoin(Paint.Join.ROUND);
      
      _overlayPaint = new Paint();
      _overlayPaint.setColor(Color.GREEN);
      _overlayPaint.setStrokeWidth(10);
      _overlayPaint.setStyle(Paint.Style.STROKE);
      _overlayPaint.setAlpha(200);
      _overlayPaint.setAntiAlias(true);
      _overlayPaint.setStrokeCap(Paint.Cap.ROUND);
      _overlayPaint.setStrokeJoin(Paint.Join.ROUND);
      
      _swipePath = new Path();
      _overlayPath = null;
      _swipePoints = new ArrayList<>();
      _keys = new HashMap<>();
      
      setBackgroundColor(Color.BLACK);
    }
    
    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh)
    {
      super.onSizeChanged(w, h, oldw, oldh);
      layoutKeys(w, h);
    }
    
    private void layoutKeys(int width, int height)
    {
      _keys.clear();
      
      // Use user's configuration for dimensions
      float keyWidth = width / 10f;
      float rowHeight = height / 3f; // 3 rows for QWERTY
      float verticalMargin = _keyVerticalMargin * rowHeight;
      float horizontalMargin = _keyHorizontalMargin * keyWidth;
      
      // Calculate text size using actual config values
      float characterSize = _characterSize;
      float labelTextSize = _labelTextSize;
      
      // Match the real keyboard's text size calculation
      float baseSize = Math.min(
        rowHeight - verticalMargin,
        (keyWidth - horizontalMargin) * 3f/2f
      );
      float textSize = baseSize * characterSize * labelTextSize;
      _textPaint.setTextSize(textSize);
      
      // Layout QWERTY keyboard
      for (int row = 0; row < KEYBOARD_LAYOUT.length; row++)
      {
        String[] rowKeys = KEYBOARD_LAYOUT[row];
        float rowOffset = row == 1 ? keyWidth * 0.5f : (row == 2 ? keyWidth : 0);
        
        for (int col = 0; col < rowKeys.length; col++)
        {
          String key = rowKeys[col];
          float x = rowOffset + col * keyWidth + horizontalMargin / 2;
          float y = row * rowHeight + verticalMargin / 2;
          
          KeyButton button = new KeyButton(key, x, y, 
            keyWidth - horizontalMargin, rowHeight - verticalMargin);
          _keys.put(key, button);
        }
      }
    }
    
    @Override
    protected void onDraw(Canvas canvas)
    {
      super.onDraw(canvas);
      
      // Draw keys
      for (KeyButton key : _keys.values())
      {
        key.draw(canvas, _keyPaint, _keyBorderPaint, _textPaint);
      }
      
      // Draw swipe path
      if (!_swipePath.isEmpty())
      {
        canvas.drawPath(_swipePath, _swipePaint);
      }
      
      // Draw overlay path (displayed after swipe completion)
      if (_overlayPath != null)
      {
        canvas.drawPath(_overlayPath, _overlayPaint);
      }
    }
    
    @Override
    public boolean onTouchEvent(MotionEvent event)
    {
      float x = event.getX();
      float y = event.getY();
      
      switch (event.getAction())
      {
        case MotionEvent.ACTION_DOWN:
          _swiping = true;
          _swipeStartTime = System.currentTimeMillis();
          _swipePath.reset();
          _swipePath.moveTo(x, y);
          _swipePoints.clear();
          _swipePoints.add(new PointF(x, y));
          invalidate();
          return true;
          
        case MotionEvent.ACTION_MOVE:
          if (_swiping)
          {
            _swipePath.lineTo(x, y);
            _swipePoints.add(new PointF(x, y));
            invalidate();
          }
          return true;
          
        case MotionEvent.ACTION_UP:
          if (_swiping)
          {
            _swiping = false;
            if (_swipePoints.size() > 5) // Minimum points for valid swipe
            {
              recordSwipe(new ArrayList<>(_swipePoints));
            }
          }
          return true;
      }
      
      return super.onTouchEvent(event);
    }
    
    public void reset()
    {
      _swipePath.reset();
      _swipePoints.clear();
      _overlayPath = null;
      invalidate();
    }
    
    public void setSwipeOverlay(Path path)
    {
      _overlayPath = path;
      invalidate();
    }
    
    public void clearOverlay()
    {
      _overlayPath = null;
      invalidate();
    }
    
    public String getKeyAt(float x, float y)
    {
      for (Map.Entry<String, KeyButton> entry : _keys.entrySet())
      {
        if (entry.getValue().contains(x, y))
        {
          return entry.getKey();
        }
      }
      return null;
    }
  }
  
  /**
   * Represents a keyboard key
   */
  private static class KeyButton
  {
    String label;
    float x, y, width, height;
    
    KeyButton(String label, float x, float y, float width, float height)
    {
      this.label = label;
      this.x = x;
      this.y = y;
      this.width = width;
      this.height = height;
    }
    
    void draw(Canvas canvas, Paint keyPaint, Paint borderPaint, Paint textPaint)
    {
      // Draw key background with rounded corners
      android.graphics.RectF rect = new android.graphics.RectF(x, y, x + width, y + height);
      float cornerRadius = Math.min(width, height) * 0.15f; // 15% of smallest dimension
      canvas.drawRoundRect(rect, cornerRadius, cornerRadius, keyPaint);
      
      // Draw border
      canvas.drawRoundRect(rect, cornerRadius, cornerRadius, borderPaint);
      
      // Draw text centered properly (accounting for text metrics)
      float textY = y + (height - textPaint.ascent() - textPaint.descent()) / 2f;
      canvas.drawText(label.toUpperCase(), x + width / 2, textY, textPaint);
    }
    
    boolean contains(float px, float py)
    {
      return px >= x && px <= x + width && py >= y && py <= y + height;
    }
  }
  
  /**
   * Represents a swipe pattern for a word
   */
  private static class SwipePattern
  {
    String word;
    List<PointF> points;
    long duration;
    
    SwipePattern(String word, List<PointF> points, long duration)
    {
      this.word = word;
      this.points = new ArrayList<>(points);
      this.duration = duration;
    }
    
    String serialize()
    {
      StringBuilder sb = new StringBuilder();
      sb.append(duration).append(",");
      for (PointF p : points)
      {
        sb.append(p.x).append(",").append(p.y).append(",");
      }
      return sb.toString();
    }
  }
}