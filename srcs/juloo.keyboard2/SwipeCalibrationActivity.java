package juloo.keyboard2;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import juloo.keyboard2.ml.SwipeMLData;
import juloo.keyboard2.ml.SwipeMLDataStore;

/**
 * Pure neural swipe calibration with ONNX transformer prediction
 * Matches web demo trace collection and animation styling
 */
public class SwipeCalibrationActivity extends Activity
{
  private static final String TAG = "NeuralCalibration";
  
  // Calibration settings
  private static final int WORDS_PER_SESSION = 20;
  
  // Test words matching web demo frequency distribution
  private static final List<String> CALIBRATION_WORDS = Arrays.asList(
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "new", "now", "old", "see", "two", "who", "boy", "did"
  );
  
  // QWERTY layout
  private static final String[][] KEYBOARD_LAYOUT = {
    {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"},
    {"a", "s", "d", "f", "g", "h", "j", "k", "l"},
    {"z", "x", "c", "v", "b", "n", "m"}
  };
  
  // UI Components
  private TextView _instructionText;
  private TextView _currentWordText;
  private TextView _progressText;
  private TextView _benchmarkText;
  private ProgressBar _progressBar;
  private NeuralKeyboardView _keyboardView;
  private Button _nextButton;
  private Button _skipButton;
  private Button _exportButton;
  private Button _benchmarkButton;
  
  // Neural prediction
  private NeuralSwipeTypingEngine _neuralEngine;
  private Config _config;
  
  // Calibration state
  private int _currentIndex = 0;
  private String _currentWord;
  private List<String> _sessionWords;
  private SwipeMLDataStore _mlDataStore;
  private int _screenWidth;
  private int _screenHeight;
  private int _keyboardHeight;
  
  // Performance tracking
  private List<Long> _predictionTimes = new ArrayList<>();
  private int _correctPredictions = 0;
  private int _totalPredictions = 0;
  
  @Override
  protected void onCreate(Bundle savedInstanceState)
  {
    super.onCreate(savedInstanceState);
    
    Log.d(TAG, "=== NEURAL CALIBRATION ACTIVITY STARTED ===");
    
    // Initialize ML data store
    _mlDataStore = SwipeMLDataStore.getInstance(this);
    
    // Initialize neural prediction engine
    SharedPreferences prefs = DirectBootAwarePreferences.get_shared_preferences(this);
    Config.initGlobalConfig(prefs, getResources(), null, false);
    _config = Config.globalConfig();
    _neuralEngine = new NeuralSwipeTypingEngine(this, _config);
    
    try
    {
      _neuralEngine.initialize();
      Log.d(TAG, "Neural engine initialized successfully");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize neural engine", e);
      showErrorDialog("Neural models failed to load. Ensure ONNX models are available in assets/models/");
      return;
    }
    
    // Get screen dimensions
    android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
    getWindowManager().getDefaultDisplay().getMetrics(metrics);
    _screenWidth = metrics.widthPixels;
    _screenHeight = metrics.heightPixels;
    _keyboardHeight = (int)(_screenHeight * 0.4f); // 40% of screen
    
    // Prepare session words
    _sessionWords = new ArrayList<>(CALIBRATION_WORDS);
    java.util.Collections.shuffle(_sessionWords);
    _sessionWords = _sessionWords.subList(0, WORDS_PER_SESSION);
    
    setupUI();
    setupKeyboard();
    showNextWord();
  }
  
  private void setupUI()
  {
    // Main layout with web demo styling
    LinearLayout mainLayout = new LinearLayout(this);
    mainLayout.setOrientation(LinearLayout.VERTICAL);
    mainLayout.setBackgroundColor(0xFF0a0a0f); // Dark background
    mainLayout.setPadding(20, 20, 20, 20);
    
    // Title
    TextView title = new TextView(this);
    title.setText("🧠 Neural Swipe Calibration");
    title.setTextSize(24);
    title.setTextColor(0xFF00d4ff); // Neon blue
    title.setPadding(0, 0, 0, 20);
    mainLayout.addView(title);
    
    // Instructions
    _instructionText = new TextView(this);
    _instructionText.setText("Swipe the word below to collect neural training data");
    _instructionText.setTextColor(0xFFFFFFFF);
    _instructionText.setPadding(0, 0, 0, 10);
    mainLayout.addView(_instructionText);
    
    // Current word display
    _currentWordText = new TextView(this);
    _currentWordText.setTextSize(32);
    _currentWordText.setTextColor(0xFFb300ff); // Neon purple
    _currentWordText.setPadding(0, 20, 0, 20);
    mainLayout.addView(_currentWordText);
    
    // Progress
    _progressText = new TextView(this);
    _progressText.setTextColor(0xFFFFFFFF);
    mainLayout.addView(_progressText);
    
    _progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
    _progressBar.setMax(WORDS_PER_SESSION);
    mainLayout.addView(_progressBar);
    
    // Benchmark display
    _benchmarkText = new TextView(this);
    _benchmarkText.setTextColor(0xFF00d4ff);
    _benchmarkText.setTextSize(14);
    _benchmarkText.setPadding(0, 10, 0, 10);
    mainLayout.addView(_benchmarkText);
    
    // Control buttons
    LinearLayout buttonLayout = new LinearLayout(this);
    buttonLayout.setOrientation(LinearLayout.HORIZONTAL);
    
    _nextButton = createNeonButton("Next", 0xFF4CAF50);
    _nextButton.setOnClickListener(v -> nextWord());
    buttonLayout.addView(_nextButton);
    
    _skipButton = createNeonButton("Skip", 0xFFFF9800);
    _skipButton.setOnClickListener(v -> skipWord());
    buttonLayout.addView(_skipButton);
    
    _exportButton = createNeonButton("Export", 0xFF2196F3);
    _exportButton.setOnClickListener(v -> exportTrainingData());
    buttonLayout.addView(_exportButton);
    
    _benchmarkButton = createNeonButton("Benchmark", 0xFFE91E63);
    _benchmarkButton.setOnClickListener(v -> runBenchmark());
    buttonLayout.addView(_benchmarkButton);
    
    Button playgroundButton = createNeonButton("🎮 Playground", 0xFF9C27B0);
    playgroundButton.setOnClickListener(v -> showNeuralPlayground());
    buttonLayout.addView(playgroundButton);
    
    mainLayout.addView(buttonLayout);
    
    setContentView(mainLayout);
  }
  
  private Button createNeonButton(String text, int color)
  {
    Button button = new Button(this);
    button.setText(text);
    button.setTextColor(0xFFFFFFFF);
    button.setBackgroundColor(color);
    button.setTextSize(14);
    LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
      0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f);
    params.setMargins(5, 5, 5, 5);
    button.setLayoutParams(params);
    return button;
  }
  
  private void setupKeyboard()
  {
    _keyboardView = new NeuralKeyboardView(this);
    _keyboardView.setLayoutParams(new ViewGroup.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT, _keyboardHeight));
    
    // Add keyboard at bottom
    ViewGroup rootLayout = (ViewGroup) findViewById(android.R.id.content).getParent();
    rootLayout.addView(_keyboardView);
    
    // Set keyboard dimensions for neural engine
    _neuralEngine.setKeyboardDimensions(_screenWidth, _keyboardHeight);
  }
  
  private void showNextWord()
  {
    if (_currentIndex >= _sessionWords.size())
    {
      showCompletionMessage();
      return;
    }
    
    _currentWord = _sessionWords.get(_currentIndex);
    _currentWordText.setText(_currentWord.toUpperCase());
    _progressText.setText(String.format("Word %d of %d", _currentIndex + 1, WORDS_PER_SESSION));
    _progressBar.setProgress(_currentIndex);
    
    updateBenchmarkDisplay();
    
    Log.d(TAG, "Showing word: " + _currentWord);
  }
  
  private void updateBenchmarkDisplay()
  {
    if (_totalPredictions > 0)
    {
      float accuracy = (_correctPredictions * 100.0f) / _totalPredictions;
      long avgTime = _predictionTimes.stream().mapToLong(Long::longValue).sum() / _predictionTimes.size();
      
      _benchmarkText.setText(String.format(
        "📊 Neural Performance: %.1f%% accuracy, %.1fms avg prediction time",
        accuracy, avgTime / 1000000.0f)); // Convert nanoseconds to milliseconds
    }
    else
    {
      _benchmarkText.setText("📊 Neural Performance: No data yet");
    }
  }
  
  private void nextWord()
  {
    _currentIndex++;
    showNextWord();
  }
  
  private void skipWord()
  {
    Log.d(TAG, "Skipped word: " + _currentWord);
    _currentIndex++;
    showNextWord();
  }
  
  private void exportTrainingData()
  {
    List<SwipeMLData> allData = _mlDataStore.loadDataBySource("neural_calibration");
    
    // Export in format matching web demo
    StringBuilder export = new StringBuilder();
    export.append("[\n");
    
    for (int i = 0; i < allData.size(); i++)
    {
      SwipeMLData data = allData.get(i);
      export.append("  {\n");
      export.append("    \"word\": \"").append(data.getTargetWord()).append("\",\n");
      export.append("    \"trajectory\": [\n");
      
      for (SwipeMLData.TracePoint point : data.getTracePoints())
      {
        export.append(String.format("      {\"x\": %.4f, \"y\": %.4f, \"t\": %d},\n",
          point.x, point.y, point.tDeltaMs));
      }
      
      export.append("    ],\n");
      export.append("    \"keys\": ").append(data.getRegisteredKeys()).append("\n");
      export.append("  }").append(i < allData.size() - 1 ? "," : "").append("\n");
    }
    
    export.append("]\n");
    
    // Copy to clipboard
    ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
    ClipData clip = ClipData.newPlainText("Neural Training Data", export.toString());
    clipboard.setPrimaryClip(clip);
    
    Toast.makeText(this, "Training data exported to clipboard", Toast.LENGTH_LONG).show();
  }
  
  private void runBenchmark()
  {
    Log.d(TAG, "Running neural prediction benchmark...");
    
    // TODO: Implement comprehensive benchmarking
    Toast.makeText(this, "Benchmark feature coming soon", Toast.LENGTH_SHORT).show();
  }
  
  private void showCompletionMessage()
  {
    _instructionText.setText("🎉 Neural Calibration Complete!");
    _currentWordText.setText("✓");
    _currentWordText.setTextColor(0xFF4CAF50);
    _progressBar.setProgress(_progressBar.getMax());
    
    updateBenchmarkDisplay();
    
    Log.d(TAG, "=== NEURAL CALIBRATION COMPLETE ===");
  }
  
  private void showNeuralPlayground()
  {
    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    builder.setTitle("🧠 Neural Parameters Playground");
    
    LinearLayout layout = new LinearLayout(this);
    layout.setOrientation(LinearLayout.VERTICAL);
    layout.setPadding(40, 20, 40, 20);
    
    // Beam width control
    addSliderControl(layout, "Beam Width", _config.neural_beam_width, 1, 16, 
      value -> _config.neural_beam_width = value);
    
    // Max length control
    addSliderControl(layout, "Max Length", _config.neural_max_length, 10, 50,
      value -> _config.neural_max_length = value);
    
    // Confidence threshold control  
    addFloatSliderControl(layout, "Confidence Threshold", _config.neural_confidence_threshold, 0.0f, 1.0f,
      value -> _config.neural_confidence_threshold = (float)value);
    
    builder.setView(layout);
    builder.setPositiveButton("Apply", (dialog, which) -> {
      _neuralEngine.setConfig(_config);
      Toast.makeText(this, "Neural parameters updated", Toast.LENGTH_SHORT).show();
    });
    builder.setNegativeButton("Cancel", null);
    builder.show();
  }
  
  private void addSliderControl(LinearLayout parent, String name, int currentValue, int min, int max,
    java.util.function.IntConsumer setter)
  {
    TextView label = new TextView(this);
    label.setText(String.format("%s: %d", name, currentValue));
    label.setTextColor(0xFFFFFFFF);
    parent.addView(label);
    
    android.widget.SeekBar slider = new android.widget.SeekBar(this);
    slider.setMax(max - min);
    slider.setProgress(currentValue - min);
    slider.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
        int value = min + progress;
        label.setText(String.format("%s: %d", name, value));
        setter.accept(value);
      }
      @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
      @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
    });
    parent.addView(slider);
  }
  
  private void addFloatSliderControl(LinearLayout parent, String name, float currentValue, float min, float max,
    java.util.function.DoubleConsumer setter)
  {
    TextView label = new TextView(this);
    label.setText(String.format("%s: %.3f", name, currentValue));
    label.setTextColor(0xFFFFFFFF);
    parent.addView(label);
    
    android.widget.SeekBar slider = new android.widget.SeekBar(this);
    slider.setMax(1000); // Fine granularity
    slider.setProgress((int)((currentValue - min) * 1000 / (max - min)));
    slider.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
        float value = min + (progress / 1000.0f) * (max - min);
        label.setText(String.format("%s: %.3f", name, value));
        setter.accept(value);
      }
      @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
      @Override public void onStopTrackingTouch(android.widget.SeekBar seekBar) {}
    });
    parent.addView(slider);
  }
  
  private void showErrorDialog(String message)
  {
    new AlertDialog.Builder(this)
      .setTitle("Neural Engine Error")
      .setMessage(message)
      .setPositiveButton("OK", (dialog, which) -> finish())
      .setCancelable(false)
      .show();
  }
  
  /**
   * Neural keyboard view with web demo styling and animations
   */
  private class NeuralKeyboardView extends View
  {
    private Map<String, KeyButton> _keys = new HashMap<>();
    private Paint _keyPaint;
    private Paint _textPaint;
    private Paint _trailPaint;
    private Paint _glowPaint;
    
    // Swipe state
    private boolean _swiping = false;
    private List<PointF> _swipePoints = new ArrayList<>();
    private List<Long> _swipeTimestamps = new ArrayList<>();
    private Path _swipePath = new Path();
    private long _swipeStartTime;
    
    public NeuralKeyboardView(Context context)
    {
      super(context);
      setupPaints();
      setOnTouchListener(this::onTouch);
    }
    
    private void setupPaints()
    {
      // Key paint - dark theme matching web demo
      _keyPaint = new Paint();
      _keyPaint.setColor(0xFF1a1a2e); // Dark key color
      _keyPaint.setAntiAlias(true);
      
      // Text paint
      _textPaint = new Paint();
      _textPaint.setColor(0xFFFFFFFF);
      _textPaint.setAntiAlias(true);
      _textPaint.setTextAlign(Paint.Align.CENTER);
      _textPaint.setTextSize(40);
      
      // Trail paint with neon glow
      _trailPaint = new Paint();
      _trailPaint.setColor(0xFF00d4ff); // Neon blue
      _trailPaint.setStrokeWidth(8);
      _trailPaint.setStyle(Paint.Style.STROKE);
      _trailPaint.setStrokeCap(Paint.Cap.ROUND);
      _trailPaint.setAntiAlias(true);
      
      // Glow paint for trail effect
      _glowPaint = new Paint();
      _glowPaint.setColor(0xFF00d4ff);
      _glowPaint.setStrokeWidth(20);
      _glowPaint.setStyle(Paint.Style.STROKE);
      _glowPaint.setStrokeCap(Paint.Cap.ROUND);
      _glowPaint.setMaskFilter(new android.graphics.BlurMaskFilter(10, android.graphics.BlurMaskFilter.Blur.NORMAL));
      _glowPaint.setAntiAlias(true);
      _glowPaint.setAlpha(100);
    }
    
    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh)
    {
      super.onSizeChanged(w, h, oldw, oldh);
      layoutKeys(w, h);
      
      // Update neural engine with keyboard dimensions and key positions
      _neuralEngine.setKeyboardDimensions(w, h);
      
      Map<Character, PointF> keyPositions = new HashMap<>();
      for (Map.Entry<String, KeyButton> entry : _keys.entrySet())
      {
        String keyStr = entry.getKey();
        if (keyStr.length() == 1)
        {
          char keyChar = keyStr.charAt(0);
          KeyButton button = entry.getValue();
          keyPositions.put(keyChar, new PointF(
            button.x + button.width/2, 
            button.y + button.height/2));
        }
      }
      _neuralEngine.setRealKeyPositions(keyPositions);
    }
    
    private void layoutKeys(int width, int height)
    {
      _keys.clear();
      
      float keyWidth = width / 10f;
      float keyHeight = height / 3f; // 3 rows
      float margin = 4;
      
      for (int row = 0; row < KEYBOARD_LAYOUT.length; row++)
      {
        String[] rowKeys = KEYBOARD_LAYOUT[row];
        float rowOffset = (row == 1) ? keyWidth * 0.5f : (row == 2) ? keyWidth : 0;
        
        for (int col = 0; col < rowKeys.length; col++)
        {
          String key = rowKeys[col];
          float x = rowOffset + col * keyWidth + margin;
          float y = row * keyHeight + margin;
          
          KeyButton button = new KeyButton(key, x, y, 
            keyWidth - margin * 2, keyHeight - margin * 2);
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
        key.draw(canvas, _keyPaint, _textPaint);
      }
      
      // Draw swipe trail with neon glow effect
      if (!_swipePath.isEmpty())
      {
        // Draw glow first
        canvas.drawPath(_swipePath, _glowPaint);
        // Draw main trail
        canvas.drawPath(_swipePath, _trailPaint);
      }
    }
    
    private boolean onTouch(View v, MotionEvent event)
    {
      float x = event.getX();
      float y = event.getY();
      
      switch (event.getAction())
      {
        case MotionEvent.ACTION_DOWN:
          startSwipe(x, y);
          return true;
          
        case MotionEvent.ACTION_MOVE:
          if (_swiping)
          {
            continueSwipe(x, y);
          }
          return true;
          
        case MotionEvent.ACTION_UP:
          if (_swiping)
          {
            endSwipe();
          }
          return true;
      }
      
      return false;
    }
    
    private void startSwipe(float x, float y)
    {
      _swiping = true;
      _swipeStartTime = System.currentTimeMillis();
      
      _swipePoints.clear();
      _swipeTimestamps.clear();
      _swipePath.reset();
      
      _swipePoints.add(new PointF(x, y));
      _swipeTimestamps.add(_swipeStartTime);
      _swipePath.moveTo(x, y);
      
      // Highlight key with glow effect
      highlightKeyAt(x, y, true);
      
      invalidate();
    }
    
    private void continueSwipe(float x, float y)
    {
      _swipePoints.add(new PointF(x, y));
      _swipeTimestamps.add(System.currentTimeMillis());
      _swipePath.lineTo(x, y);
      
      // Highlight current key
      highlightKeyAt(x, y, false);
      
      invalidate();
    }
    
    private void endSwipe()
    {
      _swiping = false;
      
      if (_swipePoints.size() > 5)
      {
        processSwipe();
      }
      
      // Clear highlights and trail after delay
      new Handler().postDelayed(() -> {
        clearHighlights();
        _swipePath.reset();
        invalidate();
      }, 500);
    }
    
    private void processSwipe()
    {
      Log.d(TAG, String.format("Processing neural swipe: %d points", _swipePoints.size()));
      
      // Create SwipeInput matching web demo format
      String keySequence = extractKeySequence();
      SwipeInput swipeInput = new SwipeInput(_swipePoints, _swipeTimestamps, new ArrayList<>());
      
      // Record ML data
      recordMLData(swipeInput);
      
      // Run neural prediction with timing
      long startTime = System.nanoTime();
      
      try
      {
        PredictionResult result = _neuralEngine.predict(swipeInput);
        long endTime = System.nanoTime();
        
        _predictionTimes.add(endTime - startTime);
        _totalPredictions++;
        
        // Check if prediction is correct
        boolean correct = false;
        for (int i = 0; i < result.words.size(); i++)
        {
          if (result.words.get(i).equals(_currentWord))
          {
            correct = true;
            _correctPredictions++;
            Log.d(TAG, String.format("✅ Correct prediction at rank %d: %s", i + 1, _currentWord));
            break;
          }
        }
        
        if (!correct)
        {
          Log.d(TAG, String.format("❌ Incorrect prediction. Expected: %s, Got: %s", 
            _currentWord, result.words.isEmpty() ? "none" : result.words.get(0)));
        }
        
        updateBenchmarkDisplay();
      }
      catch (Exception e)
      {
        Log.e(TAG, "Neural prediction failed", e);
        Toast.makeText(getContext(), "Neural prediction error: " + e.getMessage(), 
          Toast.LENGTH_SHORT).show();
      }
      
      // Auto-advance
      new Handler().postDelayed(() -> nextWord(), 1000);
    }
    
    private String extractKeySequence()
    {
      StringBuilder keySeq = new StringBuilder();
      for (PointF point : _swipePoints)
      {
        String key = getKeyAt(point.x, point.y);
        if (key != null && key.length() == 1)
        {
          keySeq.append(key);
        }
      }
      return keySeq.toString();
    }
    
    private void recordMLData(SwipeInput swipeInput)
    {
      SwipeMLData mlData = new SwipeMLData(_currentWord, "neural_calibration",
        _screenWidth, _screenHeight, _keyboardHeight);
      
      // Add trace points with timestamps
      for (int i = 0; i < _swipePoints.size(); i++)
      {
        PointF p = _swipePoints.get(i);
        long timestamp = i < _swipeTimestamps.size() ? _swipeTimestamps.get(i) : System.currentTimeMillis();
        mlData.addRawPoint(p.x, p.y, timestamp);
      }
      
      // Add registered keys
      String keySequence = extractKeySequence();
      for (char c : keySequence.toCharArray())
      {
        mlData.addRegisteredKey(String.valueOf(c));
      }
      
      _mlDataStore.storeSwipeData(mlData);
      Log.d(TAG, "Stored neural ML data for: " + _currentWord);
    }
    
    private void highlightKeyAt(float x, float y, boolean isStart)
    {
      String key = getKeyAt(x, y);
      if (key != null && _keys.containsKey(key))
      {
        KeyButton button = _keys.get(key);
        button.setHighlighted(true, isStart);
        invalidate();
      }
    }
    
    private void clearHighlights()
    {
      for (KeyButton button : _keys.values())
      {
        button.setHighlighted(false, false);
      }
    }
    
    private String getKeyAt(float x, float y)
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
   * Key button with neon styling matching web demo
   */
  private static class KeyButton
  {
    String label;
    float x, y, width, height;
    boolean highlighted = false;
    boolean isStartKey = false;
    
    KeyButton(String label, float x, float y, float width, float height)
    {
      this.label = label;
      this.x = x;
      this.y = y;
      this.width = width;
      this.height = height;
    }
    
    void setHighlighted(boolean highlighted, boolean isStart)
    {
      this.highlighted = highlighted;
      this.isStartKey = isStart;
    }
    
    void draw(Canvas canvas, Paint keyPaint, Paint textPaint)
    {
      Paint activePaint = new Paint(keyPaint);
      
      if (highlighted)
      {
        if (isStartKey)
        {
          // Start key gets special neon purple glow
          activePaint.setColor(0xFFb300ff);
          activePaint.setShadowLayer(20, 0, 0, 0xFFb300ff);
        }
        else
        {
          // Regular highlight with neon blue
          activePaint.setColor(0xFF00d4ff);
          activePaint.setShadowLayer(15, 0, 0, 0xFF00d4ff);
        }
      }
      
      // Draw key background
      canvas.drawRoundRect(x, y, x + width, y + height, 8, 8, activePaint);
      
      // Draw text
      float textY = y + height/2 - (textPaint.descent() + textPaint.ascent()) / 2;
      canvas.drawText(label.toUpperCase(), x + width/2, textY, textPaint);
    }
    
    boolean contains(float px, float py)
    {
      return px >= x && px <= x + width && py >= y && py <= y + height;
    }
  }
}