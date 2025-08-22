package juloo.keyboard2;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
  
  // Test words for calibration
  private static final List<String> CALIBRATION_WORDS = Arrays.asList(
    "the", "and", "you", "that", "this",
    "hello", "world", "thanks", "keyboard", "android",
    "swipe", "typing", "calibration", "test", "quick"
  );
  
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
  private ProgressBar _progressBar;
  private KeyboardView _keyboardView;
  private Button _nextButton;
  private Button _skipButton;
  private Button _saveButton;
  
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
  
  @Override
  protected void onCreate(Bundle savedInstanceState)
  {
    super.onCreate(savedInstanceState);
    
    Log.d(TAG, "=== CALIBRATION ACTIVITY STARTED ===");
    initializeLogging();
    
    // Initialize ML data store
    _mlDataStore = SwipeMLDataStore.getInstance(this);
    
    // Get screen dimensions
    android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
    getWindowManager().getDefaultDisplay().getMetrics(metrics);
    _screenWidth = metrics.widthPixels;
    _screenHeight = metrics.heightPixels;
    
    // Calculate keyboard height (35% of screen height, same as default keyboard)
    _keyboardHeight = (int)(_screenHeight * 0.35f);
    
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
    _instructionText.setText("Swipe the word shown below on the keyboard");
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
    
    // Buttons above keyboard
    LinearLayout buttonLayout = new LinearLayout(this);
    buttonLayout.setOrientation(LinearLayout.HORIZONTAL);
    buttonLayout.setPadding(0, 20, 0, 0);
    
    _skipButton = new Button(this);
    _skipButton.setText("Skip Word");
    _skipButton.setOnClickListener(v -> skipWord());
    buttonLayout.addView(_skipButton);
    
    _nextButton = new Button(this);
    _nextButton.setText("Next");
    _nextButton.setEnabled(false);
    _nextButton.setOnClickListener(v -> nextWord());
    buttonLayout.addView(_nextButton);
    
    _saveButton = new Button(this);
    _saveButton.setText("Save & Exit");
    _saveButton.setOnClickListener(v -> saveAndExit());
    buttonLayout.addView(_saveButton);
    
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
    
    // Shuffle and select words for this session
    List<String> allWords = new ArrayList<>(CALIBRATION_WORDS);
    Collections.shuffle(allWords);
    _sessionWords = allWords.subList(0, Math.min(WORDS_PER_SESSION, allWords.size()));
    
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
    
    // Auto-advance to next word after short delay
    Toast.makeText(this, "Swipe recorded!", Toast.LENGTH_SHORT).show();
    _keyboardView.postDelayed(() -> {
      nextWord();
    }, 800); // 800ms delay to show toast
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
    private Paint _textPaint;
    private Paint _swipePaint;
    private Map<String, KeyButton> _keys;
    private Path _swipePath;
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
      _keyPaint.setColor(Color.DKGRAY);
      _keyPaint.setStyle(Paint.Style.FILL);
      
      _textPaint = new Paint();
      _textPaint.setColor(Color.WHITE);
      _textPaint.setTextSize(40);
      _textPaint.setTextAlign(Paint.Align.CENTER);
      
      _swipePaint = new Paint();
      _swipePaint.setColor(Color.CYAN);
      _swipePaint.setStrokeWidth(8);
      _swipePaint.setStyle(Paint.Style.STROKE);
      _swipePaint.setAlpha(180);
      
      _swipePath = new Path();
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
      
      float keyWidth = width / 10f;
      float keyHeight = height / 3f;
      float padding = 4;
      
      // Layout QWERTY keyboard
      for (int row = 0; row < KEYBOARD_LAYOUT.length; row++)
      {
        String[] rowKeys = KEYBOARD_LAYOUT[row];
        float rowOffset = row == 1 ? keyWidth * 0.5f : (row == 2 ? keyWidth : 0);
        
        for (int col = 0; col < rowKeys.length; col++)
        {
          String key = rowKeys[col];
          float x = rowOffset + col * keyWidth + padding;
          float y = row * keyHeight + padding;
          
          KeyButton button = new KeyButton(key, x, y, 
            keyWidth - padding * 2, keyHeight - padding * 2);
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
      
      // Draw swipe path
      if (!_swipePath.isEmpty())
      {
        canvas.drawPath(_swipePath, _swipePaint);
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
    
    void draw(Canvas canvas, Paint keyPaint, Paint textPaint)
    {
      canvas.drawRect(x, y, x + width, y + height, keyPaint);
      canvas.drawText(label, x + width / 2, y + height / 2 + 10, textPaint);
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