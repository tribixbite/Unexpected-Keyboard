package juloo.keyboard2;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;
import juloo.keyboard2.DirectBootAwarePreferences;
import juloo.keyboard2.FoldStateTracker;
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
import android.content.ClipData;
import android.content.ClipboardManager;
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
import juloo.keyboard2.ml.SwipeMLTrainer;

/**
 * Complete swipe calibration implementation with keyboard visualization
 */
public class SwipeCalibrationActivity extends Activity
{
  private static final String TAG = "SwipeCalibration";
  private static final String PREF_KEY_CALIBRATION_DATA = "swipe_calibration_data";
  private static final String LOG_FILE = "/data/data/com.termux/files/home/calibration_log.txt";
  
  // Calibration settings
  private static final int WORDS_PER_SESSION = 20;  // Changed: 20 unique words instead of 10×2
  private static final int REPS_PER_WORD = 1;      // Changed: No repetitions
  
  // Test words for calibration - will be replaced by frequency-based selection
  private static final List<String> FALLBACK_WORDS = Arrays.asList(
    "the", "and", "you", "that", "this",
    "hello", "world", "thanks", "keyboard", "android",
    "swipe", "typing", "calibration", "test", "quick"
  );
  
  // Top frequent words from dictionary (loaded dynamically)
  private List<String> _frequentWords = new ArrayList<>();
  private Map<String, Integer> _wordFrequencies = new HashMap<>();
  
  // QWERTY layout with actual key positions - 4 rows including bottom row
  private static final String[][] KEYBOARD_LAYOUT = {
    {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"},
    {"a", "s", "d", "f", "g", "h", "j", "k", "l"},
    {"shift", "z", "x", "c", "v", "b", "n", "m", "backspace"},
    {"?123", ",", "space", ".", "enter"}
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
  private Button _exportButton;
  private Button _trainButton;
  private LinearLayout _scoreLayout;
  
  // Navigation UI components
  private LinearLayout _navigationLayout;
  private Button _prevSwipeButton;
  private Button _nextSwipeButton;
  private TextView _swipeNavText;
  private Button _deleteSwipeButton;
  private Button _browseSwipesButton;
  
  // Real-time accuracy metrics components
  private LinearLayout _metricsLayout;
  private TextView _sessionAccuracyText;
  private TextView _overallAccuracyText;
  private TextView _wpmText;
  private TextView _confusionPatternsText;
  private ProgressBar _accuracyProgressBar;
  
  // Calibration state
  private int _currentIndex = 0;
  // private int _currentRep = 0;  // No longer used - we have 20 unique words instead
  private String _currentWord;
  private List<String> _sessionWords;
  private Map<String, List<SwipePattern>> _calibrationData;
  private BufferedWriter _logWriter;
  private List<PointF> _currentSwipePoints;
  private List<Long> _currentSwipeTimestamps;
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
  
  // Swipe navigation state
  private List<SwipeMLData> _allSwipes;
  private int _currentSwipeIndex = -1;
  private boolean _isBrowsingMode = false;
  
  // ML Training
  private SwipeMLTrainer _mlTrainer;
  
  // Template comparison features
  private WordGestureTemplateGenerator _templateGenerator;
  private TextView _templateComparisonText;
  private Button _copyComparisonButton;
  private StringBuilder _comparisonData;
  
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
    // Note: Don't load calibration data here as DTW predictor needs keyboard dimensions first
    
    // Initialize UI handler for delayed operations
    _uiHandler = new Handler();
    
    // Load frequency dictionary
    loadFrequencyDictionary();
    
    // Get screen dimensions
    android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
    getWindowManager().getDefaultDisplay().getMetrics(metrics);
    _screenWidth = metrics.widthPixels;
    _screenHeight = metrics.heightPixels;
    
    // Load user's keyboard configuration from proper device-protected preferences
    // Use DirectBootAwarePreferences to match the main keyboard's storage location
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
    
    // Debug: Log all preference keys and values
    Log.d(TAG, "=== DEBUG: Reading preferences ===");
    Log.d(TAG, "Preference keys found: " + prefs.getAll().keySet());
    Log.d(TAG, "All preferences: " + prefs.getAll());
    
    // Try to read keyboard_height in different ways
    try {
      int kh = prefs.getInt("keyboard_height", -1);
      Log.d(TAG, "keyboard_height as int: " + kh);
    } catch (Exception e) {
      Log.d(TAG, "Failed to read keyboard_height as int: " + e.getMessage());
      try {
        String kh = prefs.getString("keyboard_height", "null");
        Log.d(TAG, "keyboard_height as string: " + kh);
      } catch (Exception e2) {
        Log.d(TAG, "Failed to read keyboard_height as string: " + e2.getMessage());
      }
    }
    
    // Check for foldable device state
    FoldStateTracker foldTracker = new FoldStateTracker(this);
    boolean foldableUnfolded = foldTracker.isUnfolded();
    Log.d(TAG, "Foldable device unfolded: " + foldableUnfolded);
    
    // Get keyboard height percentage from user settings
    // Check for landscape mode
    boolean isLandscape = getResources().getConfiguration().orientation == 
                          android.content.res.Configuration.ORIENTATION_LANDSCAPE;
    Log.d(TAG, "Is landscape: " + isLandscape);
    
    int keyboardHeightPref;
    if (isLandscape)
    {
      // Check for foldable unfolded state first, then regular landscape
      String key = foldableUnfolded ? "keyboard_height_landscape_unfolded" : "keyboard_height_landscape";
      keyboardHeightPref = prefs.getInt(key, 50);
      Log.d(TAG, "Reading landscape height from key '" + key + "': " + keyboardHeightPref);
    }
    else
    {
      // Check for foldable unfolded state first, then regular portrait
      String key = foldableUnfolded ? "keyboard_height_unfolded" : "keyboard_height";
      keyboardHeightPref = prefs.getInt(key, 35);
      Log.d(TAG, "Reading portrait height from key '" + key + "': " + keyboardHeightPref);
    }
    
    // Calculate keyboard height
    float keyboardHeightPercent = keyboardHeightPref / 100.0f;
    _keyboardHeight = (int)(_screenHeight * keyboardHeightPercent);
    Log.d(TAG, "Calculated keyboard height: " + _keyboardHeight + " pixels (" + keyboardHeightPref + "% of " + _screenHeight + ")");
    
    // Load user's text and margin settings
    _characterSize = prefs.getFloat("character_size", 1.15f);
    _labelTextSize = 0.33f; // Default label text size
    _keyVerticalMargin = prefs.getFloat("key_vertical_margin", 1.5f) / 100;
    
    _keyHorizontalMargin = prefs.getFloat("key_horizontal_margin", 2.0f) / 100;
    
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
    _progressBar.setMax(WORDS_PER_SESSION);  // Total words = 20
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
    
    _exportButton = new Button(this);
    _exportButton.setText("Export to Clipboard");
    _exportButton.setOnClickListener(v -> exportToClipboard());
    buttonLayout.addView(_exportButton);
    
    _trainButton = new Button(this);
    _trainButton.setText("Train Model");
    _trainButton.setOnClickListener(v -> startTraining());
    buttonLayout.addView(_trainButton);
    
    _browseSwipesButton = new Button(this);
    _browseSwipesButton.setText("Browse Swipes");
    _browseSwipesButton.setOnClickListener(v -> enterBrowseMode());
    buttonLayout.addView(_browseSwipesButton);
    
    topLayout.addView(buttonLayout);
    
    // Navigation controls (initially hidden)
    _navigationLayout = new LinearLayout(this);
    _navigationLayout.setOrientation(LinearLayout.HORIZONTAL);
    _navigationLayout.setPadding(10, 10, 10, 10);
    _navigationLayout.setGravity(android.view.Gravity.CENTER);
    _navigationLayout.setVisibility(View.GONE);
    
    _prevSwipeButton = new Button(this);
    _prevSwipeButton.setText("◀ Previous");
    _prevSwipeButton.setOnClickListener(v -> navigatePreviousSwipe());
    _navigationLayout.addView(_prevSwipeButton);
    
    _swipeNavText = new TextView(this);
    _swipeNavText.setText("Swipe 0 of 0");
    _swipeNavText.setTextColor(Color.WHITE);
    _swipeNavText.setPadding(20, 0, 20, 0);
    _navigationLayout.addView(_swipeNavText);
    
    _nextSwipeButton = new Button(this);
    _nextSwipeButton.setText("Next ▶");
    _nextSwipeButton.setOnClickListener(v -> navigateNextSwipe());
    _navigationLayout.addView(_nextSwipeButton);
    
    _deleteSwipeButton = new Button(this);
    _deleteSwipeButton.setText("Delete This");
    _deleteSwipeButton.setOnClickListener(v -> deleteCurrentSwipe());
    _navigationLayout.addView(_deleteSwipeButton);
    
    Button exitBrowse = new Button(this);
    exitBrowse.setText("Exit Browse");
    exitBrowse.setOnClickListener(v -> exitBrowseMode());
    _navigationLayout.addView(exitBrowse);
    
    topLayout.addView(_navigationLayout);
    mainLayout.addView(topLayout);
    
    // Keyboard at bottom with real dimensions
    _keyboardView = new KeyboardView(this);
    android.widget.RelativeLayout.LayoutParams keyboardParams = new android.widget.RelativeLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT, _keyboardHeight);
    keyboardParams.addRule(android.widget.RelativeLayout.ALIGN_PARENT_BOTTOM);
    _keyboardView.setLayoutParams(keyboardParams);
    mainLayout.addView(_keyboardView);
    
    // Template comparison section
    TextView comparisonLabel = new TextView(this);
    comparisonLabel.setText("Template vs User Gesture Comparison:");
    comparisonLabel.setTextSize(16);
    comparisonLabel.setPadding(16, 16, 16, 8);
    mainLayout.addView(comparisonLabel);
    
    _templateComparisonText = new TextView(this);
    _templateComparisonText.setText("Swipe words to see template comparison data...");
    _templateComparisonText.setTextSize(12);
    _templateComparisonText.setPadding(16, 0, 16, 8);
    _templateComparisonText.setMaxLines(10);
    _templateComparisonText.setVerticalScrollBarEnabled(true);
    android.widget.ScrollView scrollView = new android.widget.ScrollView(this);
    scrollView.addView(_templateComparisonText);
    scrollView.setLayoutParams(new LinearLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT, 200));
    mainLayout.addView(scrollView);
    
    _copyComparisonButton = new Button(this);
    _copyComparisonButton.setText("Copy Comparison Data");
    _copyComparisonButton.setOnClickListener(v -> copyComparisonData());
    mainLayout.addView(_copyComparisonButton);
    
    setContentView(mainLayout);
    
    // Initialize template generator for comparison
    _templateGenerator = new WordGestureTemplateGenerator();
    _templateGenerator.loadDictionary(this);
    _comparisonData = new StringBuilder();
    
    // Initialize ML trainer
    _mlTrainer = new SwipeMLTrainer(this);
    _mlTrainer.setTrainingListener(new TrainingListenerImpl());
    
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
    // Use device-protected storage for weights to be consistent
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
    // Use device-protected storage for weights to be consistent
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
    _currentSwipeTimestamps = new ArrayList<>();
    _sessionStartTime = System.currentTimeMillis();
    
    // Load overall accuracy from preferences
    loadOverallAccuracy();
    
    // Select words from top 30% most frequent
    Log.d(TAG, "Selecting calibration words. Dictionary size: " + _frequentWords.size());
    if (_frequentWords.isEmpty())
    {
      // Fallback if dictionary didn't load
      Log.w(TAG, "Dictionary is empty! Using fallback words");
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
    // _currentRep = 0;  // No longer used
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
    
    int totalProgress = _currentIndex;  // No reps, just word index
    _progressBar.setProgress(totalProgress);
    _progressText.setText(String.format("Word %d of %d",
      _currentIndex + 1, _sessionWords.size()));  // Simplified: no rep count
    
    _nextButton.setEnabled(false);
    _keyboardView.reset();
    
    logMessage("Showing word: " + _currentWord + " (word " + (_currentIndex + 1) + " of " + _sessionWords.size() + ")");
  }
  
  private void recordSwipe(List<PointF> points)
  {
    if (points.isEmpty()) return;
    
    long duration = System.currentTimeMillis() - _swipeStartTime;
    
    // Show the swipe path overlay
    showSwipePathOverlay(points);
    
    // ADD: Template comparison analysis for debugging
    addTemplateComparison(_currentWord, points);
    
    // Create swipe pattern for legacy storage
    SwipePattern pattern = new SwipePattern(_currentWord, points, duration);
    
    // Add to calibration data
    if (!_calibrationData.containsKey(_currentWord))
    {
      _calibrationData.put(_currentWord, new ArrayList<>());
    }
    _calibrationData.get(_currentWord).add(pattern);
    
    // Create ML data object with actual keyboard dimensions
    SwipeMLData mlData = new SwipeMLData(_currentWord, "calibration",
                                         _screenWidth, _screenHeight, _keyboardHeight);
    
    // Add trace points with actual timestamps in keyboard-relative coordinates
    for (int i = 0; i < points.size() && i < _currentSwipeTimestamps.size(); i++)
    {
      PointF p = points.get(i);
      long timestamp = _currentSwipeTimestamps.get(i);
      
      // Store keyboard-relative coordinates (already relative from KeyboardView)
      mlData.addRawPoint(p.x, p.y, timestamp);
      
      // Add registered key
      String key = _keyboardView.getKeyAt(p.x, p.y);
      if (key != null)
      {
        mlData.addRegisteredKey(key);
      }
    }
    
    // Add keyboard dimensions metadata (keyboard is at bottom, so Y is screenHeight - keyboardHeight)
    int keyboardY = _screenHeight - _keyboardHeight;
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
    // No repetitions - just move to next word
    _currentIndex++;
    showNextWord();
  }
  
  private void skipWord()
  {
    logMessage("Skipped word: " + _currentWord);
    // No repetitions - just move to next word
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
    // Use device-protected storage for calibration data
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
        new InputStreamReader(getAssets().open("dictionaries/en_enhanced.txt")));
      
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
      if (_frequentWords.size() > 0)
      {
        Log.d(TAG, "First 20 frequent words: " + _frequentWords.subList(0, Math.min(20, _frequentWords.size())));
      }
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load dictionary: " + e.getMessage(), e);
      e.printStackTrace();
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
   * Export calibration data to clipboard for debugging - now uses ML data store
   */
  private void exportToClipboard()
  {
    try
    {
      // Load all calibration swipes from ML data store
      List<SwipeMLData> allSwipes = _mlDataStore.loadDataBySource("calibration");
      
      if (allSwipes.isEmpty())
      {
        Toast.makeText(this, "No calibration data to export", Toast.LENGTH_SHORT).show();
        return;
      }
      
      // Create comprehensive JSON export
      StringBuilder export = new StringBuilder();
      export.append("{\n");
      export.append("  \"timestamp\": ").append(System.currentTimeMillis()).append(",\n");
      export.append("  \"generated\": \"").append(new Date().toString()).append("\",\n");
      export.append("  \"total_swipes\": ").append(allSwipes.size()).append(",\n");
      export.append("  \"keyboard_height\": ").append(_keyboardHeight).append(",\n");
      export.append("  \"screen_width\": ").append(_screenWidth).append(",\n");
      export.append("  \"screen_height\": ").append(_screenHeight).append(",\n");
      export.append("  \"algorithm_weights\": {\n");
      export.append("    \"dtw\": ").append(_dtwWeight).append(",\n");
      export.append("    \"gaussian\": ").append(_gaussianWeight).append(",\n");
      export.append("    \"ngram\": ").append(_ngramWeight).append(",\n");
      export.append("    \"frequency\": ").append(_frequencyWeight).append("\n");
      export.append("  },\n");
      export.append("  \"swipe_data\": [\n");
      
      boolean first = true;
      for (SwipeMLData swipe : allSwipes)
      {
        if (!first) export.append(",\n");
        
        export.append("    {\n");
        export.append("      \"word\": \"").append(swipe.getTargetWord()).append("\",\n");
        export.append("      \"trace_id\": \"").append(swipe.getTraceId()).append("\",\n");
        export.append("      \"timestamp_utc\": ").append(swipe.getTimestampUtc()).append(",\n");
        export.append("      \"source\": \"").append(swipe.getCollectionSource()).append("\",\n");
        
        List<SwipeMLData.TracePoint> points = swipe.getTracePoints();
        export.append("      \"point_count\": ").append(points.size()).append(",\n");
        export.append("      \"duration_ms\": ").append(calculateDuration(points)).append(",\n");
        export.append("      \"registered_keys\": ").append(swipe.getRegisteredKeys().toString()).append(",\n");
        export.append("      \"normalized_trace\": [\n");
        
        boolean firstPoint = true;
        for (SwipeMLData.TracePoint tp : points)
        {
          if (!firstPoint) export.append(",\n");
          export.append("        {\"x\": ").append(tp.x)
                .append(", \"y\": ").append(tp.y)
                .append(", \"t_delta_ms\": ").append(tp.tDeltaMs).append("}");
          firstPoint = false;
        }
        
        export.append("\n      ]\n");
        export.append("    }");
        first = false;
      }
      
      export.append("\n  ],\n");
      export.append("  \"accuracy_metrics\": {\n");
      export.append("    \"session_correct\": ").append(_sessionCorrectCount).append(",\n");
      export.append("    \"session_total\": ").append(_sessionTotalCount).append(",\n");
      export.append("    \"overall_correct\": ").append(_overallCorrectCount).append(",\n");
      export.append("    \"overall_total\": ").append(_overallTotalCount).append("\n");
      export.append("  },\n");
      
      // Add summary statistics
      Map<String, Integer> wordCounts = new HashMap<>();
      int totalPoints = 0;
      long totalDuration = 0;
      
      for (SwipeMLData swipe : allSwipes)
      {
        String word = swipe.getTargetWord();
        wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
        
        List<SwipeMLData.TracePoint> points = swipe.getTracePoints();
        totalPoints += points.size();
        totalDuration += calculateDuration(points);
      }
      
      export.append("  \"summary_stats\": {\n");
      export.append("    \"unique_words\": ").append(wordCounts.size()).append(",\n");
      export.append("    \"avg_points_per_swipe\": ").append(totalPoints / allSwipes.size()).append(",\n");
      export.append("    \"avg_duration_ms\": ").append(totalDuration / allSwipes.size()).append(",\n");
      export.append("    \"words_practiced\": ").append(wordCounts.toString()).append("\n");
      export.append("  }\n");
      export.append("}");
      
      // Copy to clipboard
      ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
      ClipData clip = ClipData.newPlainText("Swipe Calibration Data", export.toString());
      clipboard.setPrimaryClip(clip);
      
      // Show success message with summary
      String message = String.format("Exported %d swipes (%d words) to clipboard as JSON", 
                                    allSwipes.size(), wordCounts.size());
      Toast.makeText(this, message, Toast.LENGTH_LONG).show();
      logMessage("ML calibration data exported: " + allSwipes.size() + " swipes, " + export.length() + " bytes");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to export calibration data: " + e.getMessage(), e);
      Toast.makeText(this, "Export failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
    }
  }
  
  /**
   * Delete all stored calibration samples
   */
  private void deleteStoredSamples()
  {
    // Clear ML data store
    _mlDataStore.clearAllData();
    
    // Clear SharedPreferences from device-protected storage
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
    
    // Ensure DTW predictor is properly initialized before use
    if (_keyboardView.getWidth() > 0 && _keyboardView.getHeight() > 0)
    {
      _dtwPredictor.setKeyboardDimensions(_keyboardView.getWidth(), _keyboardView.getHeight());
      _dtwPredictor.loadCalibrationData(this);
    }
    
    // Get touched keys from keyboard view with position data
    List<KeyboardData.Key> touchedKeys = new ArrayList<>();
    Map<String, KeyButton> keyPositions = _keyboardView.getKeyPositions();
    
    for (PointF p : points)
    {
      String keyChar = _keyboardView.getKeyAt(p.x, p.y);
      if (keyChar != null && keyPositions.containsKey(keyChar))
      {
        KeyButton keyButton = keyPositions.get(keyChar);
        // Create a Key object with position data for DTW predictor
        KeyValue kv = KeyValue.makeStringKey(keyChar);
        // Store position in shift and width for DTW to use
        KeyboardData.Key key = new KeyboardData.Key(
          new KeyValue[]{kv, null, null, null, null},
          null, // anticircle
          0,    // flags
          keyButton.width / _keyboardView.getWidth(), // normalized width
          keyButton.x / _keyboardView.getWidth(), // normalized x position in shift
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
    // Use device-protected storage for metrics
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
    // Use device-protected storage for metrics
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
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
      float rowHeight = height / 4f; // 4 rows now including bottom row
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
      
      // Layout QWERTY keyboard with 4 rows
      for (int row = 0; row < KEYBOARD_LAYOUT.length; row++)
      {
        String[] rowKeys = KEYBOARD_LAYOUT[row];
        
        if (row == 0) // Top row - numbers row (q-p)
        {
          for (int col = 0; col < rowKeys.length; col++)
          {
            String key = rowKeys[col];
            float x = col * keyWidth + horizontalMargin / 2;
            float y = row * rowHeight + verticalMargin / 2;
            
            KeyButton button = new KeyButton(key, x, y, 
              keyWidth - horizontalMargin, rowHeight - verticalMargin);
            _keys.put(key, button);
          }
        }
        else if (row == 1) // Second row (a-l) - offset by half key
        {
          float rowOffset = keyWidth * 0.5f;
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
        else if (row == 2) // Third row (shift, z-m, backspace)
        {
          float currentX = horizontalMargin / 2;
          for (int col = 0; col < rowKeys.length; col++)
          {
            String key = rowKeys[col];
            float keyW = keyWidth - horizontalMargin;
            
            // Special keys are wider
            if (key.equals("shift") || key.equals("backspace"))
            {
              keyW = keyWidth * 1.5f - horizontalMargin;
            }
            
            float y = row * rowHeight + verticalMargin / 2;
            
            KeyButton button = new KeyButton(key, currentX, y, 
              keyW, rowHeight - verticalMargin);
            _keys.put(key, button);
            
            currentX += keyW + horizontalMargin;
          }
        }
        else if (row == 3) // Bottom row (?123, comma, space, period, enter)
        {
          float currentX = horizontalMargin / 2;
          for (int col = 0; col < rowKeys.length; col++)
          {
            String key = rowKeys[col];
            float keyW = keyWidth - horizontalMargin;
            
            // Special key widths
            if (key.equals("space"))
            {
              keyW = keyWidth * 5f - horizontalMargin; // Space bar is 5 keys wide
            }
            else if (key.equals("?123") || key.equals("enter"))
            {
              keyW = keyWidth * 1.5f - horizontalMargin;
            }
            
            float y = row * rowHeight + verticalMargin / 2;
            
            KeyButton button = new KeyButton(key, currentX, y, 
              keyW, rowHeight - verticalMargin);
            _keys.put(key, button);
            
            currentX += keyW + horizontalMargin;
          }
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
          _currentSwipeTimestamps.clear();
          _currentSwipeTimestamps.add(System.currentTimeMillis());
          invalidate();
          return true;
          
        case MotionEvent.ACTION_MOVE:
          if (_swiping)
          {
            _swipePath.lineTo(x, y);
            _swipePoints.add(new PointF(x, y));
            _currentSwipeTimestamps.add(System.currentTimeMillis());
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
            _currentSwipeTimestamps.clear();
          }
          return true;
      }
      
      return super.onTouchEvent(event);
    }
    
    public void reset()
    {
      _swipePath.reset();
      _swipePoints.clear();
      _currentSwipeTimestamps.clear();
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
    
    public Map<String, KeyButton> getKeyPositions()
    {
      return new HashMap<>(_keys);
    }
    
    public void displaySwipeTrace(List<PointF> points)
    {
      if (points == null || points.isEmpty()) return;
      
      _overlayPath = new Path();
      _overlayPath.moveTo(points.get(0).x, points.get(0).y);
      
      for (int i = 1; i < points.size(); i++)
      {
        _overlayPath.lineTo(points.get(i).x, points.get(i).y);
      }
      
      invalidate();
    }
    
    public void clearSwipeOverlay()
    {
      _overlayPath = null;
      invalidate();
    }
  }
  
  /**
   * Represents a keyboard key
   */
  static class KeyButton
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
      String displayLabel = label;
      
      // Handle special key labels
      if (label.equals("space"))
      {
        displayLabel = " "; // Space bar typically has no label or just a space
      }
      else if (label.equals("shift"))
      {
        displayLabel = "⇧"; // Shift arrow symbol
      }
      else if (label.equals("backspace"))
      {
        displayLabel = "⌫"; // Backspace symbol
      }
      else if (label.equals("enter"))
      {
        displayLabel = "↵"; // Enter/return symbol
      }
      else if (label.equals("?123"))
      {
        displayLabel = "?123"; // Keep as is
      }
      else
      {
        displayLabel = label.toUpperCase(); // Regular keys in uppercase
      }
      
      float textY = y + (height - textPaint.ascent() - textPaint.descent()) / 2f;
      canvas.drawText(displayLabel, x + width / 2, textY, textPaint);
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
  
  // Helper methods for trace point conversion
  private List<PointF> convertToNormalizedPoints(List<SwipeMLData.TracePoint> tracePoints)
  {
    List<PointF> points = new ArrayList<>();
    for (SwipeMLData.TracePoint tp : tracePoints)
    {
      points.add(new PointF(tp.x, tp.y));  // Use public fields, not getters
    }
    return points;
  }
  
  private long calculateDuration(List<SwipeMLData.TracePoint> tracePoints)
  {
    if (tracePoints.isEmpty()) return 0;
    
    // Duration is sum of all deltaMs values
    long duration = 0;
    for (SwipeMLData.TracePoint tp : tracePoints)
    {
      duration += tp.tDeltaMs;  // Use public field, not getter
    }
    return duration;
  }
  
  // Navigation methods for browsing recorded swipes
  private void enterBrowseMode()
  {
    _isBrowsingMode = true;
    
    // Load all swipes from database
    _allSwipes = _mlDataStore.loadDataBySource("calibration");
    
    if (_allSwipes.isEmpty())
    {
      Toast.makeText(this, "No recorded swipes to browse", Toast.LENGTH_SHORT).show();
      return;
    }
    
    // Show navigation controls, hide normal controls
    _navigationLayout.setVisibility(View.VISIBLE);
    _nextButton.setEnabled(false);
    _skipButton.setEnabled(false);
    _instructionText.setText("Browsing recorded swipes - use arrows to navigate");
    
    // Start at first swipe
    _currentSwipeIndex = 0;
    displaySwipeAtIndex(0);
    
    Log.d(TAG, "Entered browse mode with " + _allSwipes.size() + " swipes");
  }
  
  private void exitBrowseMode()
  {
    _isBrowsingMode = false;
    _navigationLayout.setVisibility(View.GONE);
    _nextButton.setEnabled(true);
    _skipButton.setEnabled(true);
    _instructionText.setText("Swipe the word shown below - auto-advances on completion");
    
    // Clear displayed trace
    _keyboardView.clearSwipeOverlay();
    
    // Return to normal calibration
    showNextWord();
  }
  
  private void navigatePreviousSwipe()
  {
    if (_currentSwipeIndex > 0)
    {
      _currentSwipeIndex--;
      displaySwipeAtIndex(_currentSwipeIndex);
    }
    else
    {
      Toast.makeText(this, "At first swipe", Toast.LENGTH_SHORT).show();
    }
  }
  
  private void navigateNextSwipe()
  {
    if (_currentSwipeIndex < _allSwipes.size() - 1)
    {
      _currentSwipeIndex++;
      displaySwipeAtIndex(_currentSwipeIndex);
    }
    else
    {
      Toast.makeText(this, "At last swipe", Toast.LENGTH_SHORT).show();
    }
  }
  
  private void displaySwipeAtIndex(int index)
  {
    if (index < 0 || index >= _allSwipes.size()) return;
    
    SwipeMLData data = _allSwipes.get(index);
    
    // Update navigation text
    _swipeNavText.setText(String.format("Swipe %d of %d", index + 1, _allSwipes.size()));
    
    // Show target word
    _currentWordText.setText(data.getTargetWord());
    
    // Convert normalized points to screen points and display trace
    List<PointF> screenPoints = new ArrayList<>();
    List<PointF> normalizedPoints = convertToNormalizedPoints(data.getTracePoints());
    
    for (PointF p : normalizedPoints)
    {
      // Convert from normalized [0,1] to keyboard coordinates
      float screenX = p.x * _keyboardView.getWidth();
      float screenY = p.y * _keyboardView.getHeight();
      screenPoints.add(new PointF(screenX, screenY));
    }
    
    // Display the trace on keyboard
    _keyboardView.displaySwipeTrace(screenPoints);
    
    // Show metadata
    long duration = calculateDuration(data.getTracePoints());
    int pointCount = normalizedPoints.size();
    _scoreText.setText(String.format("Duration: %dms, Points: %d", duration, pointCount));
    _scoreLayout.setVisibility(View.VISIBLE);
    
    // Update navigation button states
    _prevSwipeButton.setEnabled(index > 0);
    _nextSwipeButton.setEnabled(index < _allSwipes.size() - 1);
  }
  
  private void deleteCurrentSwipe()
  {
    if (_currentSwipeIndex < 0 || _currentSwipeIndex >= _allSwipes.size()) return;
    
    SwipeMLData toDelete = _allSwipes.get(_currentSwipeIndex);
    
    // Confirm deletion
    new AlertDialog.Builder(this)
      .setTitle("Delete Swipe")
      .setMessage("Delete this swipe for '" + toDelete.getTargetWord() + "'?")
      .setPositiveButton("Delete", (dialog, which) -> {
        // Remove from database
        _mlDataStore.deleteEntry(toDelete);
        
        // Remove from list
        _allSwipes.remove(_currentSwipeIndex);
        
        // Adjust index and display
        if (_allSwipes.isEmpty())
        {
          exitBrowseMode();
          Toast.makeText(this, "No more swipes to browse", Toast.LENGTH_SHORT).show();
        }
        else
        {
          if (_currentSwipeIndex >= _allSwipes.size())
          {
            _currentSwipeIndex = _allSwipes.size() - 1;
          }
          displaySwipeAtIndex(_currentSwipeIndex);
        }
        
        Toast.makeText(this, "Swipe deleted", Toast.LENGTH_SHORT).show();
      })
      .setNegativeButton("Cancel", null)
      .show();
  }
  
  private void startTraining()
  {
    if (_mlTrainer.isTraining())
    {
      Toast.makeText(this, "Training already in progress...", Toast.LENGTH_SHORT).show();
      return;
    }
    
    if (!_mlTrainer.canTrain())
    {
      Toast.makeText(this, "Need at least 100 samples to train. Record more swipes first.", Toast.LENGTH_LONG).show();
      return;
    }
    
    // Disable train button during training
    _trainButton.setEnabled(false);
    _trainButton.setText("Training...");
    
    // Start training
    _mlTrainer.startTraining();
  }
  
  private class TrainingListenerImpl implements SwipeMLTrainer.TrainingListener
  {
    @Override
    public void onTrainingStarted()
    {
      runOnUiThread(() -> {
        Toast.makeText(SwipeCalibrationActivity.this, "Training started...", Toast.LENGTH_SHORT).show();
        logMessage("ML training started");
      });
    }
    
    @Override
    public void onTrainingProgress(int progress, int total)
    {
      runOnUiThread(() -> {
        _trainButton.setText(String.format("Training... %d%%", progress));
        logMessage("Training progress: " + progress + "%");
      });
    }
    
    @Override
    public void onTrainingCompleted(SwipeMLTrainer.TrainingResult result)
    {
      runOnUiThread(() -> {
        _trainButton.setEnabled(true);
        _trainButton.setText("Train Model");
        
        String message = String.format("Training completed!\n• %d samples used\n• Training time: %dms\n• Accuracy: %.1f%%", 
                                       result.samplesUsed, result.trainingTimeMs, result.accuracy * 100);
        
        new AlertDialog.Builder(SwipeCalibrationActivity.this)
          .setTitle("Training Complete")
          .setMessage(message)
          .setPositiveButton("OK", null)
          .show();
        
        logMessage("Training completed: " + result.samplesUsed + " samples, " + 
                  result.trainingTimeMs + "ms, accuracy=" + result.accuracy);
        
        // Apply training results to improve predictions
        applyTrainingResults(result);
      });
    }
    
    @Override
    public void onTrainingError(String error)
    {
      runOnUiThread(() -> {
        _trainButton.setEnabled(true);
        _trainButton.setText("Train Model");
        
        Toast.makeText(SwipeCalibrationActivity.this, "Training failed: " + error, Toast.LENGTH_LONG).show();
        logMessage("Training error: " + error);
      });
    }
  }
  
  private void applyTrainingResults(SwipeMLTrainer.TrainingResult result)
  {
    try
    {
      // Store original weights for comparison
      float originalDtwWeight = _dtwWeight;
      float originalGaussianWeight = _gaussianWeight;
      float originalNgramWeight = _ngramWeight;
      float originalFrequencyWeight = _frequencyWeight;
      
      // If training achieved good accuracy, adjust weights to favor the model
      if (result.accuracy >= 0.8f)
      {
        // Increase DTW weight as it's our primary algorithm for now
        _dtwWeight = Math.min(0.6f, _dtwWeight + 0.1f);
        _gaussianWeight = Math.max(0.2f, _gaussianWeight - 0.05f);
        
        // Update UI
        _dtwWeightSlider.setProgress(Math.round(_dtwWeight * 100));
        _gaussianWeightSlider.setProgress(Math.round(_gaussianWeight * 100));
        
        normalizeWeights();
        
        // Calculate actual changes
        float dtwChange = _dtwWeight - originalDtwWeight;
        float gaussianChange = _gaussianWeight - originalGaussianWeight;
        float ngramChange = _ngramWeight - originalNgramWeight;
        float frequencyChange = _frequencyWeight - originalFrequencyWeight;
        
        // Create detailed visual feedback
        StringBuilder changeMessage = new StringBuilder();
        changeMessage.append("🎯 Algorithm Weights Updated!\n\n");
        
        if (Math.abs(dtwChange) > 0.001f)
        {
          changeMessage.append(String.format("📈 DTW: %.1f%% → %.1f%% (%+.1f%%)\n", 
                                           originalDtwWeight * 100, _dtwWeight * 100, dtwChange * 100));
        }
        if (Math.abs(gaussianChange) > 0.001f)
        {
          changeMessage.append(String.format("📊 Gaussian: %.1f%% → %.1f%% (%+.1f%%)\n", 
                                           originalGaussianWeight * 100, _gaussianWeight * 100, gaussianChange * 100));
        }
        if (Math.abs(ngramChange) > 0.001f)
        {
          changeMessage.append(String.format("📝 N-gram: %.1f%% → %.1f%% (%+.1f%%)\n", 
                                           originalNgramWeight * 100, _ngramWeight * 100, ngramChange * 100));
        }
        if (Math.abs(frequencyChange) > 0.001f)
        {
          changeMessage.append(String.format("📊 Frequency: %.1f%% → %.1f%% (%+.1f%%)\n", 
                                           originalFrequencyWeight * 100, _frequencyWeight * 100, frequencyChange * 100));
        }
        
        changeMessage.append(String.format("\n✨ Training accuracy: %.1f%%", result.accuracy * 100));
        changeMessage.append("\n💡 These optimizations should improve prediction accuracy!");
        
        // Show detailed dialog
        new AlertDialog.Builder(this)
          .setTitle("Weights Optimization Applied")
          .setMessage(changeMessage.toString())
          .setPositiveButton("Great!", null)
          .setIcon(android.R.drawable.ic_dialog_info)
          .show();
        
        logMessage("Applied training results: DTW " + String.format("%+.3f", dtwChange) + 
                  ", Gaussian " + String.format("%+.3f", gaussianChange) + 
                  ", N-gram " + String.format("%+.3f", ngramChange) + 
                  ", Frequency " + String.format("%+.3f", frequencyChange));
      }
      else
      {
        // Show why weights weren't changed
        String lowAccuracyMessage = String.format(
          "⚠️ Training Accuracy Too Low\n\n" +
          "Accuracy achieved: %.1f%%\n" +
          "Minimum required: 80.0%%\n\n" +
          "💡 Try recording more diverse swipe samples to improve training quality.\n\n" +
          "Algorithm weights remain unchanged:",
          result.accuracy * 100);
          
        lowAccuracyMessage += String.format(
          "\n• DTW: %.1f%%\n• Gaussian: %.1f%%\n• N-gram: %.1f%%\n• Frequency: %.1f%%",
          _dtwWeight * 100, _gaussianWeight * 100, _ngramWeight * 100, _frequencyWeight * 100);
        
        new AlertDialog.Builder(this)
          .setTitle("Weights Not Changed")
          .setMessage(lowAccuracyMessage)
          .setPositiveButton("OK", null)
          .setIcon(android.R.drawable.ic_dialog_alert)
          .show();
        
        logMessage("Training accuracy too low to apply results: " + result.accuracy);
      }
      
      // Store training metadata for future reference
      SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
      prefs.edit()
        .putLong("last_training_time", System.currentTimeMillis())
        .putInt("last_training_samples", result.samplesUsed)
        .putFloat("last_training_accuracy", result.accuracy)
        .putString("last_model_version", result.modelVersion)
        .apply();
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to apply training results", e);
      Toast.makeText(this, "Failed to apply training results: " + e.getMessage(), Toast.LENGTH_LONG).show();
    }
  }
  
  /**
   * Copy comparison data to clipboard
   */
  private void copyComparisonData()
  {
    ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
    ClipData clip = ClipData.newPlainText("Template Comparison Data", _comparisonData.toString());
    clipboard.setPrimaryClip(clip);
    Toast.makeText(this, "Comparison data copied to clipboard", Toast.LENGTH_SHORT).show();
  }
  
  /**
   * Add template comparison data for a swiped word
   */
  private void addTemplateComparison(String word, List<PointF> userSwipe)
  {
    try
    {
      // Generate template for this word
      ContinuousGestureRecognizer.Template template = _templateGenerator.generateWordTemplate(word);
      if (template == null)
      {
        Log.w(TAG, "No template generated for word: " + word);
        return;
      }
      
      // DEBUGGING: Check template before any processing
      Log.d(TAG, "Generated template for " + word + ": " + template.pts.size() + " points");
      if (template.pts.size() >= 2)
      {
        ContinuousGestureRecognizer.Point first = template.pts.get(0);
        ContinuousGestureRecognizer.Point last = template.pts.get(template.pts.size() - 1);
        Log.d(TAG, "Raw template: (" + first.x + "," + first.y + ") → (" + last.x + "," + last.y + ")");
      }
      
      // Convert user swipe to CGR format and normalize to template coordinate space
      List<ContinuousGestureRecognizer.Point> userPoints = new ArrayList<>();
      
      // Get keyboard dimensions for coordinate transformation
      float keyboardWidth = _keyboardView.getWidth();
      float keyboardHeight = _keyboardView.getHeight();
      
      for (PointF p : userSwipe)
      {
        // Transform user coordinates from screen space to template space (0-1000)
        double normalizedX = (p.x / keyboardWidth) * 1000.0;
        double normalizedY = (p.y / keyboardHeight) * 1000.0;
        userPoints.add(new ContinuousGestureRecognizer.Point(normalizedX, normalizedY));
      }
      
      // FULL CGR TESTING: Use same 3000 template set as normal keyboard
      ContinuousGestureRecognizer cgr = new ContinuousGestureRecognizer();
      
      // Generate FULL template set (same as normal keyboard) 
      List<ContinuousGestureRecognizer.Template> fullTemplates = 
        _templateGenerator.generateBalancedWordTemplates(3000);
      
      android.util.Log.d(TAG, "Testing CGR with FULL " + fullTemplates.size() + " templates (same as normal keyboard)");
      
      cgr.setTemplateSet(fullTemplates);
      List<ContinuousGestureRecognizer.Result> results = cgr.recognize(userPoints);
      
      // Build detailed comparison data
      StringBuilder comparison = new StringBuilder();
      comparison.append("=== WORD: ").append(word.toUpperCase()).append(" ===\n");
      comparison.append("Template points: ").append(template.pts.size()).append("\n");
      comparison.append("User swipe points: ").append(userSwipe.size()).append("\n");
      
      // Template shape analysis
      if (template.pts.size() >= 2)
      {
        ContinuousGestureRecognizer.Point first = template.pts.get(0);
        ContinuousGestureRecognizer.Point last = template.pts.get(template.pts.size() - 1);
        double templateLength = calculatePathLength(template.pts);
        comparison.append("Template: (").append(String.format("%.0f", first.x))
                  .append(",").append(String.format("%.0f", first.y))
                  .append(") → (").append(String.format("%.0f", last.x))
                  .append(",").append(String.format("%.0f", last.y))
                  .append(") len=").append(String.format("%.0f", templateLength)).append("\n");
      }
      
      // User swipe analysis
      if (userSwipe.size() >= 2)
      {
        PointF first = userSwipe.get(0);
        PointF last = userSwipe.get(userSwipe.size() - 1);
        double userLength = calculatePathLength(userSwipe);
        comparison.append("User: (").append(String.format("%.0f", first.x))
                  .append(",").append(String.format("%.0f", first.y))
                  .append(") → (").append(String.format("%.0f", last.x))
                  .append(",").append(String.format("%.0f", last.y))
                  .append(") len=").append(String.format("%.0f", userLength)).append("\n");
      }
      
      // CGR recognition results with match calculations
      comparison.append("CGR RESULTS:\n");
      for (int i = 0; i < Math.min(3, results.size()); i++)
      {
        ContinuousGestureRecognizer.Result result = results.get(i);
        comparison.append(String.format("  #%d: %s (prob=%.6f)\n", 
                         i + 1, result.template.id, result.prob));
      }
      
      if (results.isEmpty())
      {
        comparison.append("  No CGR recognition results\n");
      }
      
      // Match point calculation analysis
      boolean correctMatch = !results.isEmpty() && results.get(0).template.id.equals(word);
      comparison.append("MATCH ANALYSIS:\n");
      comparison.append("  Correct recognition: ").append(correctMatch ? "YES" : "NO").append("\n");
      if (!results.isEmpty())
      {
        comparison.append("  Confidence level: ");
        double prob = results.get(0).prob;
        if (prob > 0.5) comparison.append("HIGH");
        else if (prob > 0.1) comparison.append("MEDIUM"); 
        else if (prob > 0.01) comparison.append("LOW");
        else comparison.append("VERY LOW");
        comparison.append(String.format(" (%.6f)\n", prob));
      }
      
      comparison.append("Timestamp: ").append(new Date().toString()).append("\n");
      comparison.append("----------------------------------------\n\n");
      
      // Add to accumulated data
      _comparisonData.append(comparison);
      
      // Update display (show last 3 comparisons)
      String[] entries = _comparisonData.toString().split("----------------------------------------");
      StringBuilder display = new StringBuilder();
      int start = Math.max(0, entries.length - 3);
      for (int i = start; i < entries.length; i++)
      {
        if (!entries[i].trim().isEmpty())
        {
          display.append(entries[i]).append("----------------------------------------\n");
        }
      }
      
      _templateComparisonText.setText(display.toString());
      
    }
    catch (Exception e)
    {
      Log.e(TAG, "Error in template comparison: " + e.getMessage());
      _templateComparisonText.setText("Error: " + e.getMessage());
    }
  }
  
  /**
   * Calculate path length for comparison (handles both PointF and CGR Point types)
   */
  private double calculatePathLength(List<?> points)
  {
    if (points.size() < 2) return 0;
    
    double length = 0;
    for (int i = 1; i < points.size(); i++)
    {
      double dx, dy;
      if (points.get(0) instanceof PointF)
      {
        PointF p1 = (PointF) points.get(i - 1);
        PointF p2 = (PointF) points.get(i);
        dx = p2.x - p1.x;
        dy = p2.y - p1.y;
      }
      else
      {
        ContinuousGestureRecognizer.Point p1 = (ContinuousGestureRecognizer.Point) points.get(i - 1);
        ContinuousGestureRecognizer.Point p2 = (ContinuousGestureRecognizer.Point) points.get(i);
        dx = p2.x - p1.x;
        dy = p2.y - p1.y;
      }
      length += Math.sqrt(dx * dx + dy * dy);
    }
    return length;
  }
}