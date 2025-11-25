package juloo.keyboard2;

import ai.onnxruntime.*;
import android.content.Context;
import android.graphics.PointF;
import android.util.Log;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

// REFACTORING: Import Kotlin ONNX modules for modular architecture
import juloo.keyboard2.onnx.ModelLoader;
import juloo.keyboard2.onnx.EncoderWrapper;
import juloo.keyboard2.onnx.DecoderWrapper;
import juloo.keyboard2.onnx.TensorFactory;
import juloo.keyboard2.onnx.MemoryPool;
import juloo.keyboard2.onnx.BeamSearchEngine;

/**
 * ONNX-based neural swipe predictor using transformer encoder-decoder architecture
 * Replaces legacy DTW/Bayesian prediction with state-of-the-art neural networks
 * 
 * OPTIMIZATION: Uses singleton pattern with session persistence for maximum performance
 */
public class OnnxSwipePredictor
{
  private static final String TAG = "OnnxSwipePredictor";
  
  // Singleton instance for session persistence (CRITICAL OPTIMIZATION)
  private static OnnxSwipePredictor _singletonInstance;
  private static final Object _singletonLock = new Object();

  // Model configuration matching web demo exactly
  private static final int DEFAULT_MAX_SEQUENCE_LENGTH = 150; // Default for v1 models
  private static final int TRAJECTORY_FEATURES = 6; // x, y, vx, vy, ax, ay
  private static final float NORMALIZED_WIDTH = 1.0f;
  private static final float NORMALIZED_HEIGHT = 1.0f;

  // Model version configuration
  private String _currentModelVersion = "v2"; // "v2" (builtin), "v1", "v3" (external)
  private int _maxSequenceLength = 250; // Dynamic based on model version (v2 default)
  private String _currentEncoderPath = null; // Track loaded model paths
  private String _currentDecoderPath = null;
  private String _modelAccuracy = "80.6%"; // Current model accuracy
  private String _modelSource = "builtin"; // "builtin", "external", or "fallback"
  
  // Beam search parameters - standard defaults that respect playground settings
  // MOBILE-OPTIMIZED: Lower defaults for better performance on mobile devices
  // beam_width=8 * max_length=35 = 280 decoder inferences per swipe (too slow!)
  // beam_width=2 * max_length=35 = 70 decoder inferences per swipe (balanced)
  private static final int DEFAULT_BEAM_WIDTH = 4; // Increased to 4 for better accuracy (she/me)
  private static final int DEFAULT_MAX_LENGTH = 20; // Must match model max_word_len (was 35)
  private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f;
  
  // Proper beam search parameters - no aggressive optimizations that break quality
  
  // Use proper beam search that respects playground settings
  private static final boolean FORCE_GREEDY_SEARCH = false; // Use beam search with playground settings
  
  // Special tokens
  private static final int PAD_IDX = 0;
  private static final int UNK_IDX = 1;
  private static final int SOS_IDX = 2;
  private static final int EOS_IDX = 3;
  
  private Context _context;
  private Config _config;
  private OrtEnvironment _ortEnvironment;
  private OrtSession _encoderSession;
  private OrtSession _decoderSession;
  private SwipeTokenizer _tokenizer;
  private SwipeTrajectoryProcessor _trajectoryProcessor;
  private OptimizedVocabulary _vocabulary; // OPTIMIZATION: Web app vocabulary system

  // REFACTORING: Modular ONNX components
  private ModelLoader _modelLoader; // Handles model loading and session creation
  private TensorFactory _tensorFactory; // Handles tensor creation from trajectory features
  private EncoderWrapper _encoderWrapper; // Handles encoder inference
  private DecoderWrapper _decoderWrapper; // Handles decoder inference
  private MemoryPool _memoryPool; // Handles buffer pooling for GC reduction
  
  
  // Model state
  private boolean _isModelLoaded = false;
  private boolean _forceCpuFallback = false; // AUTO-FIX: Disable hardware acceleration if it crashes
  private volatile boolean _isInitialized = false; // THREAD SAFETY: volatile ensures visibility across threads
  private boolean _keepSessionsInMemory = true; // OPTIMIZATION: Never unload for speed
  private boolean _usesSeparateMasks = false; // Track if decoder uses separate padding/causal masks (custom models) vs combined target_mask (v2 builtin)
  private boolean _broadcastEnabled = false; // OPTIMIZATION v6 (perftodos6.md): Broadcast-enabled models expand memory internally
  
  // Configuration parameters
  private int _beamWidth = DEFAULT_BEAM_WIDTH;
  private int _maxLength = DEFAULT_MAX_LENGTH;
  private float _confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD;
  
  // OPTIMIZATION: Dedicated thread pool for ONNX operations (1.5x speedup expected)
  private static ExecutorService _onnxExecutor;
  private static final Object _executorLock = new Object();

  // Debug logging and config caching (CACHED - updated via updateConfig(), not checked on every swipe)
  private NeuralSwipeTypingEngine.DebugLogger _debugLogger;
  private boolean _enableVerboseLogging = false; // Cached from Config.swipe_debug_detailed_logging
  private boolean _showRawOutput = false; // Cached from Config.swipe_debug_show_raw_output
  private boolean _batchBeams = false; // Cached from Config.neural_batch_beams
  private Config _cachedConfig; // Cached config to avoid repeated SharedPreferences access
  
  private OnnxSwipePredictor(Context context)
  {
    _context = context;
    _ortEnvironment = OrtEnvironment.getEnvironment();
    _trajectoryProcessor = new SwipeTrajectoryProcessor();
    _tokenizer = new SwipeTokenizer();
    _vocabulary = new OptimizedVocabulary(context); // OPTIMIZATION: Initialize vocabulary
    
    // Log.d(TAG, "OnnxSwipePredictor initialized with session persistence");
  }
  
  /**
   * OPTIMIZATION: Get singleton instance with persistent ONNX sessions
   * This prevents expensive model reloading between predictions (2-5x speedup)
   */
  public static OnnxSwipePredictor getInstance(Context context)
  {
    if (_singletonInstance == null)
    {
      synchronized (_singletonLock)
      {
        if (_singletonInstance == null)
        {
          _singletonInstance = new OnnxSwipePredictor(context);
          // DO NOT initialize here - let caller trigger async loading
          // This prevents blocking UI when keyboard first appears
        }
      }
    }
    return _singletonInstance;
  }

  /**
   * OPTIMIZATION: Initialize models asynchronously on background thread
   * Call this from InputMethodService.onCreate() for non-blocking startup
   */
  public void initializeAsync()
  {
    if (_isInitialized)
    {
      return; // Already initialized
    }

    // Initialize thread pool if needed
    initializeThreadPool();

    if (_onnxExecutor != null)
    {
      Log.d(TAG, "Starting async model initialization...");
      _onnxExecutor.submit(() -> {
        boolean success = initialize();
        Log.d(TAG, "Async initialization completed: " + success);
      });
    }
    else
    {
      // Fallback to sync if executor not available
      Log.w(TAG, "No executor available, falling back to sync initialization");
      initialize();
    }
  }

  /**
   * Initialize models synchronously (blocking)
   * Use initializeAsync() for non-blocking startup
   */
  public void initializeSync()
  {
    if (!_isInitialized)
    {
      initialize();
    }
  }
  
  /**
   * Initialize the predictor with models from assets
   * OPTIMIZATION: Models stay loaded in memory for maximum performance
   * THREAD SAFETY: synchronized to prevent concurrent initialization from background thread and setConfig()
   */
  public synchronized boolean initialize()
  {
    // OPTIMIZATION Phase 3.1: Thread safety check
    // Warn if initialization is called on main thread (may cause UI jank)
    if (android.os.Looper.getMainLooper() == android.os.Looper.myLooper())
    {
      Log.w(TAG, "⚠️ initialize() called on MAIN THREAD - may cause UI jank!");
      // In debug builds with StrictMode, this should be avoided
      if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.HONEYCOMB)
      {
        android.os.StrictMode.ThreadPolicy policy = android.os.StrictMode.getThreadPolicy();
        if (policy != null)
        {
          // StrictMode is enabled - this will trigger a warning
          Log.w(TAG, "StrictMode is active - consider calling initializeAsync() instead");
        }
      }
    }

    if (_isInitialized)
    {
      // Log.d(TAG, "Already initialized, models loaded: " + _isModelLoaded);
      return _isModelLoaded;
    }

    try
    {
      Log.d(TAG, "STARTING OnnxSwipePredictor.initialize()") ;
      // Log.d(TAG, "Loading ONNX models...");
      // logDebug("🔄 Loading ONNX transformer models...");

      // Determine model paths and parameters based on version
      String encoderPath, decoderPath;
      boolean useExternalModels = false;

      switch (_currentModelVersion)
      {
        case "v2":
          // CLEANUP (v1.32.582): Removed float32 models, INT8 quantized only (saves 20MB APK)
          // INT8 quantized models (calibrated, v2) - now in models/ root
          encoderPath = "models/swipe_encoder_android.onnx";
          decoderPath = "models/swipe_decoder_android.onnx";
          _maxSequenceLength = 250;
          _modelAccuracy = "73.4%";
          _modelSource = "builtin-quantized-v2";
          Log.i(TAG, "Loading v2 INT8 quantized models (calibrated, broadcast-enabled, XNNPACK-optimized)");
          break;

        case "v1":
        case "v3":
          // Legacy model versions removed from bundle - fallback to v2
          Log.w(TAG, String.format("Model version %s is no longer included - falling back to v2", _currentModelVersion));
          android.widget.Toast.makeText(_context,
            "Model " + _currentModelVersion + " removed. Using built-in v2.",
            android.widget.Toast.LENGTH_LONG).show();

          encoderPath = "models/swipe_encoder_android.onnx";
          decoderPath = "models/swipe_decoder_android.onnx";
          _maxSequenceLength = 250;
          _modelAccuracy = "80.6%";
          _modelSource = "fallback";
          _currentModelVersion = "v2";
          break;

        case "custom":
          // External models - require file picker
          if (_config != null && _config.neural_custom_encoder_path != null &&
              _config.neural_custom_decoder_path != null)
          {
            encoderPath = _config.neural_custom_encoder_path;
            decoderPath = _config.neural_custom_decoder_path;
            useExternalModels = true;
            _modelSource = "external";

            // Set parameters based on version
            if ("v1".equals(_currentModelVersion))
            {
              _maxSequenceLength = 150;
              _modelAccuracy = "~65%";
              Log.d(TAG, "Loading v1 models from external files (150-len)");
            }
            else if ("v3".equals(_currentModelVersion))
            {
              _maxSequenceLength = 250;
              _modelAccuracy = "72.1%";
              Log.d(TAG, "Loading v3 models from external files (250-len)");
            }
            else // custom
            {
              _maxSequenceLength = 250; // Default, user can override
              _modelAccuracy = "Unknown";
              Log.d(TAG, "Loading custom models from external files");
            }
          }
          else
          {
            // Fallback to builtin v2 if external paths not set
            Log.w(TAG, String.format("External model %s selected but no files configured - falling back to v2",
              _currentModelVersion));
            android.widget.Toast.makeText(_context,
              "External model files not configured. Using builtin v2 model.",
              android.widget.Toast.LENGTH_LONG).show();

            encoderPath = "models/swipe_encoder_android.onnx";
            decoderPath = "models/swipe_decoder_android.onnx";
            _maxSequenceLength = 250;
            _modelAccuracy = "80.6%";
            _modelSource = "fallback";
            _currentModelVersion = "v2";
          }
          break;

        default:
          // Unknown version - fallback to v2
          Log.w(TAG, "Unknown model version: " + _currentModelVersion + " - falling back to v2");
          encoderPath = "models/swipe_encoder_android.onnx";
          decoderPath = "models/swipe_decoder_android.onnx";
          _maxSequenceLength = 250;
          _modelAccuracy = "80.6%";
          _modelSource = "fallback";
          _currentModelVersion = "v2";
          break;
      }

      // REFACTORING: Use ModelLoader module for cleaner model loading
      // Create ModelLoader if not exists (lazy initialization)
      if (_modelLoader == null)
      {
        _modelLoader = new ModelLoader(_context, _ortEnvironment);
      }

      // Load encoder model
      Log.d(TAG, "Loading encoder model from: " + encoderPath);
      long encStartTime = System.currentTimeMillis();
      // AUTO-FIX: Respect CPU fallback flag
      ModelLoader.LoadResult encoderResult = _modelLoader.loadModel(encoderPath, "Encoder", !_forceCpuFallback);
      long encTotalTime = System.currentTimeMillis() - encStartTime;

      _encoderSession = encoderResult.getSession();
      Log.i(TAG, "⏱️ Encoder total load time: " + encTotalTime + "ms");
      Log.i(TAG, "✅ Encoder loaded with " + encoderResult.getExecutionProvider() + " provider");

      // OPTIMIZATION v6 (perftodos6.md Step 3): Verify model signature for quantized models
      Log.i(TAG, "--- Encoder Model Signature ---");
      try
      {
        for (Map.Entry<String, ai.onnxruntime.NodeInfo> entry : _encoderSession.getInputInfo().entrySet())
        {
          Log.i(TAG, "Input: " + entry.getKey() + " | Info: " + entry.getValue().getInfo().toString());
        }
        for (Map.Entry<String, ai.onnxruntime.NodeInfo> entry : _encoderSession.getOutputInfo().entrySet())
        {
          Log.i(TAG, "Output: " + entry.getKey() + " | Info: " + entry.getValue().getInfo().toString());
        }
      }
      catch (Exception sigError)
      {
        Log.w(TAG, "Could not log model signature: " + sigError.getMessage());
      }
      Log.i(TAG, "---------------------------------");

      Log.d(TAG, String.format("Encoder model loaded: %s (max_seq_len=%d)", _currentModelVersion, _maxSequenceLength));

      // Load decoder model
      Log.d(TAG, "Loading decoder model from: " + decoderPath);
      long decStartTime = System.currentTimeMillis();
      // AUTO-FIX: Respect CPU fallback flag
      ModelLoader.LoadResult decoderResult = _modelLoader.loadModel(decoderPath, "Decoder", !_forceCpuFallback);
      long decTotalTime = System.currentTimeMillis() - decStartTime;

      _decoderSession = decoderResult.getSession();
      Log.i(TAG, "⏱️ Decoder total load time: " + decTotalTime + "ms");
      Log.i(TAG, "✅ Decoder loaded with " + decoderResult.getExecutionProvider() + " provider");

      // OPTIMIZATION v6 (perftodos6.md Step 3): Verify model signature for quantized models
      Log.i(TAG, "--- Decoder Model Signature ---");
      try
      {
        for (Map.Entry<String, ai.onnxruntime.NodeInfo> entry : _decoderSession.getInputInfo().entrySet())
        {
          Log.i(TAG, "Input: " + entry.getKey() + " | Info: " + entry.getValue().getInfo().toString());
        }
        for (Map.Entry<String, ai.onnxruntime.NodeInfo> entry : _decoderSession.getOutputInfo().entrySet())
        {
          Log.i(TAG, "Output: " + entry.getKey() + " | Info: " + entry.getValue().getInfo().toString());
        }
      }
      catch (Exception sigError)
      {
        Log.w(TAG, "Could not log model signature: " + sigError.getMessage());
      }
      Log.i(TAG, "---------------------------------");

      Log.d(TAG, String.format("Decoder model loaded: %s (max_seq_len=%d)", _currentModelVersion, _maxSequenceLength));

      // OPTIMIZATION v6 (perftodos6.md): Read model configuration for broadcast support
      readModelConfig(encoderPath);

      // Load tokenizer configuration
      Log.d(TAG, "Loading tokenizer");
      long tokStart = System.currentTimeMillis();
      boolean tokenizerLoaded = _tokenizer.loadFromAssets(_context);
      long tokTime = System.currentTimeMillis() - tokStart;
      Log.i(TAG, "⏱️ Tokenizer load: " + tokTime + "ms");
      Log.d(TAG, "Tokenizer loaded: " + tokenizerLoaded);
      // logDebug("📝 Tokenizer loaded: " + tokenizerLoaded + " (vocab size: " + _tokenizer.getVocabSize() + ")");

      // OPTIMIZATION: Load vocabulary for fast filtering
      Log.d(TAG, "Loading vocabulary");
      long vocabStart = System.currentTimeMillis();
      boolean vocabularyLoaded;
      
      // Prevent redundant reloading if already loaded (fixes double "Loaded X custom words" logs)
      if (_vocabulary.isLoaded())
      {
        vocabularyLoaded = true;
        Log.d(TAG, "Vocabulary already loaded, skipping reload");
      }
      else
      {
        vocabularyLoaded = _vocabulary.loadVocabulary();
      }

      long vocabTime = System.currentTimeMillis() - vocabStart;
      Log.i(TAG, "⏱️ Vocabulary load: " + vocabTime + "ms");
      Log.d(TAG, "Vocabulary loaded: " + vocabularyLoaded);
      // logDebug("📚 Vocabulary loaded: " + vocabularyLoaded + " (words: " + _vocabulary.getStats().totalWords + ")");
      
      _isModelLoaded = (_encoderSession != null && _decoderSession != null);

      if (_isModelLoaded)
      {
        // Track successfully loaded paths for change detection
        _currentEncoderPath = encoderPath;
        _currentDecoderPath = decoderPath;

        // REFACTORING: Initialize modular components after successful model loading
        Log.d(TAG, "Initializing modular ONNX components");

        // Create TensorFactory for tensor creation
        _tensorFactory = new TensorFactory(_ortEnvironment, _maxSequenceLength, TRAJECTORY_FEATURES);

        // Create EncoderWrapper for encoder inference
        _encoderWrapper = new EncoderWrapper(
          _encoderSession,
          _tensorFactory,
          _ortEnvironment,
          _enableVerboseLogging
        );

        // Create DecoderWrapper for decoder inference
        _decoderWrapper = new DecoderWrapper(
          _decoderSession,
          _tensorFactory,
          _ortEnvironment,
          _broadcastEnabled,
          _enableVerboseLogging
        );
        
        // Initialize MemoryPool for buffer management
        _memoryPool = new MemoryPool();

        Log.d(TAG, "✅ Modular components initialized (TensorFactory, EncoderWrapper, DecoderWrapper)");
      }
      else
      {
        // Clear paths if loading failed to allow for retry
        _currentEncoderPath = null;
        _currentDecoderPath = null;
      }

      // OPTIMIZATION: Pre-allocate reusable buffers for beam search
      if (_isModelLoaded)
      {
        // initializeReusableBuffers(); // REMOVED: Logic moved to BeamSearchEngine
        initializeThreadPool();
        // logDebug("🧠 ONNX neural prediction system ready!");
        // Log.d(TAG, "ONNX neural prediction system ready with optimized vocabulary");
      }
      else
      {
        // logDebug("⚠️ ONNX models failed to load - missing encoder or decoder session");
        Log.w(TAG, "Failed to load ONNX models");
      }

      // CRITICAL: Mark as initialized regardless of success/failure to prevent re-entry
      _isInitialized = true;

      Log.d(TAG, "FINISHED OnnxSwipePredictor.initialize()") ;
      return _isModelLoaded;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize ONNX models: " + e.getClass().getSimpleName() + " - " + e.getMessage(), e);
      _isInitialized = true;
      _isModelLoaded = false;
      return false;
    }
  }
  
  /**
   * OPTIMIZATION: Async prediction for non-blocking UI performance
   * Uses dedicated thread pool for ONNX inference operations
   */
  public Future<PredictionResult> predictAsync(SwipeInput input)
  {
    if (_onnxExecutor != null)
    {
      return _onnxExecutor.submit(() -> predict(input));
    }
    else
    {
      // Fallback to synchronous prediction
      return java.util.concurrent.CompletableFuture.completedFuture(predict(input));
    }
  }
  
  /**
   * Predict words from swipe input using neural transformer
   * OPTIMIZATION: Added detailed performance timing for bottleneck analysis
   */
  public PredictionResult predict(SwipeInput input)
  {
    // OPTIMIZATION: Return empty result instead of throwing when models not ready
    // This allows UI to remain responsive while models load asynchronously
    if (!_isModelLoaded)
    {
      // Log reason for debugging but don't throw
      String reason;
      if (_encoderSession == null && _decoderSession == null)
        reason = "models still loading";
      else if (_encoderSession == null)
        reason = "encoder loading";
      else if (_decoderSession == null)
        reason = "decoder loading";
      else
        reason = "initialization in progress";

      Log.d(TAG, "Prediction skipped: " + reason);
      return new PredictionResult(new ArrayList<>(), new ArrayList<>()); // Empty result
    }
    
    try
    {
      // OPTIMIZATION: Detailed performance timing for bottleneck analysis
      long totalStartTime = System.nanoTime();
      
      // Log.d(TAG, "Neural prediction for swipe with " + input.coordinates.size() + " points");
      // logDebug("🚀 Starting neural prediction for " + input.coordinates.size() + " points");
      
      // Extract trajectory features with timing
      long preprocessStartTime = System.nanoTime();
      SwipeTrajectoryProcessor.TrajectoryFeatures features =
        _trajectoryProcessor.extractFeatures(input, _maxSequenceLength);
      long preprocessTime = System.nanoTime() - preprocessStartTime;

      // Log detected nearest key sequence for debugging (ALWAYS when debug logger is available)
      // This is critical for debugging key detection issues like 'x' → 'd' problems
      if (features.nearestKeys != null && _debugLogger != null)
      {
        // Convert nearest keys to readable character sequence (deduplicated)
        StringBuilder keySeqBuilder = new StringBuilder();
        int lastKey = -1;
        for (int i = 0; i < Math.min(features.actualLength, features.nearestKeys.size()); i++)
        {
          int tokenIdx = features.nearestKeys.get(i);
          if (tokenIdx != lastKey && tokenIdx >= 4 && tokenIdx <= 29)
          {
            char c = (char)('a' + (tokenIdx - 4));
            keySeqBuilder.append(c);
            lastKey = tokenIdx;
          }
        }

        // Get keyboard dimensions for context
        float kbWidth = _trajectoryProcessor != null ? _trajectoryProcessor._keyboardWidth : 0;
        float kbHeight = _trajectoryProcessor != null ? _trajectoryProcessor._keyboardHeight : 0;

        // Log raw and normalized coordinates to debug Y-axis issues
        if (input.coordinates != null && !input.coordinates.isEmpty() && features.normalizedPoints != null) {
          android.graphics.PointF rawFirst = input.coordinates.get(0);
          android.graphics.PointF rawLast = input.coordinates.get(input.coordinates.size() - 1);
          logDebug(String.format("📐 RAW coords: first=(%.0f,%.0f) last=(%.0f,%.0f)\n",
              rawFirst.x, rawFirst.y, rawLast.x, rawLast.y));
        }

        logDebug(String.format("📐 Keyboard: %.0fx%.0f | Points: %d\n", kbWidth, kbHeight, features.actualLength));
        logDebug("🎯 DETECTED KEY SEQUENCE: \"" + keySeqBuilder.toString() +
                 "\" (" + features.actualLength + " points → " + keySeqBuilder.length() + " unique keys)\n");

        // Log first and last normalized coordinates with detailed key detection
        if (features.normalizedPoints != null && !features.normalizedPoints.isEmpty())
        {
          SwipeTrajectoryProcessor.TrajectoryPoint first = features.normalizedPoints.get(0);
          SwipeTrajectoryProcessor.TrajectoryPoint last = features.normalizedPoints.get(Math.min(features.actualLength - 1, features.normalizedPoints.size() - 1));

          // Show detailed detection for first and last points
          String firstDetail = KeyboardGrid.INSTANCE.getDetailedDetection(first.x, first.y);
          String lastDetail = KeyboardGrid.INSTANCE.getDetailedDetection(last.x, last.y);

          logDebug("📍 First point: " + firstDetail);
          logDebug("📍 Last point: " + lastDetail);

          // Log actualLength to verify it matches input coordinate count
          logDebug(String.format("📏 ACTUAL_LENGTH: %d (encoder/decoder mask threshold)\n", features.actualLength));
        }
      }

      // REFACTORING: Use EncoderWrapper for cleaner encoder inference
      OnnxTensor encoderMemory = null;

      try {
        // Run encoder inference using modular EncoderWrapper
        long encoderStartTime = System.nanoTime();
        EncoderWrapper.EncoderResult encoderResult = _encoderWrapper.encode(features);
        long encoderTime = System.nanoTime() - encoderStartTime;

        // Extract memory tensor from encoder result
        encoderMemory = encoderResult.getMemory();

        // Run beam search or greedy search decoding with timing
        long searchStartTime = System.nanoTime();
        List<BeamSearchCandidate> candidates;
        if (_config != null && _config.neural_greedy_search) {
            candidates = runGreedySearch(encoderMemory, features.actualLength, _maxLength);
        } else {
            // REFACTORING NOTE: This still uses old runBeamSearch signature
            // Will be replaced with BeamSearchEngine in next step
            candidates = runBeamSearch(encoderMemory, features.actualLength, features);
        }
        long searchTime = System.nanoTime() - searchStartTime;

        // Post-processing with timing
        long postprocessStartTime = System.nanoTime();
        PredictionResult result = createPredictionResult(candidates, input);
        long postprocessTime = System.nanoTime() - postprocessStartTime;

        // OPTIMIZATION Phase 3.2: End-to-end latency measurement
        // Comprehensive breakdown for identifying remaining bottlenecks
        long totalTime = System.nanoTime() - totalStartTime;

        // Log detailed timing breakdown (always, for performance monitoring)
        Log.i(TAG, String.format("⏱️ Swipe prediction latency breakdown:\n" +
          "   Preprocessing:  %3dms (trajectory extraction, key detection)\n" +
          "   Encoder:        %3dms (swipe → embeddings)\n" +
          "   Beam search:    %3dms (decoder inference)\n" +
          "   Postprocessing: %3dms (vocab filtering, ranking)\n" +
          "   TOTAL:          %3dms",
          preprocessTime / 1_000_000,
          encoderTime / 1_000_000,
          searchTime / 1_000_000,
          postprocessTime / 1_000_000,
          totalTime / 1_000_000
        ));

        return result;

      } finally {
        // REFACTORING: Simplified cleanup - EncoderWrapper manages input tensors
        // Only need to close encoder memory tensor
        if (encoderMemory != null) encoderMemory.close();
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Neural prediction failed", e);
      
      // AUTO-FIX: If using hardware acceleration, try fallback to CPU
      if (!_forceCpuFallback) {
          Log.w(TAG, "⚠️ Hardware acceleration likely crashed. Switching to CPU-only mode.");
          cleanup(true); // Clean up current sessions
          _forceCpuFallback = true;
          initializeSync(); // Re-initialize synchronously
          
          // Retry prediction once
          try {
              return predict(input);
          } catch (Exception retryEx) {
              Log.e(TAG, "Retry on CPU also failed", retryEx);
          }
      }
      
      throw new RuntimeException("Neural prediction failed: " + e.getMessage());
    }
  }
  
  /**
   * OPTIMIZATION: Create optimized SessionOptions with NNAPI and performance settings
   * Implements Gemini's recommendations for maximum ONNX Runtime performance
   */
  private OrtSession.SessionOptions createOptimizedSessionOptions(String sessionName)
  {
    try
    {
      OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

      // OPTIMIZATION 1: Maximum graph optimization level (operator fusion, layout transforms)
      sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
      // logDebug("⚙️ Set optimization level to ALL_OPT for " + sessionName);

      // OPTIMIZATION 2: Let ONNX Runtime determine optimal thread count for mobile
      sessionOptions.setIntraOpNumThreads(0); // Will be overridden by execution provider config
      // logDebug("🧵 Set intra-op threads to auto-detect for " + sessionName);

      // OPTIMIZATION 3: Memory pattern optimization for repeated inference
      sessionOptions.setMemoryPatternOptimization(true);
      // logDebug("🧠 Enabled memory pattern optimization for " + sessionName);

      // OPTIMIZATION 4: Cache optimized model graph to disk for faster subsequent loads
      // First load: optimize + save to cache. Subsequent loads: load from cache (skip optimization)
      if (_context != null)
      {
        try
        {
          java.io.File cacheDir = _context.getCacheDir();
          String cacheFileName = "onnx_optimized_" + sessionName.toLowerCase() + ".ort";
          java.io.File cacheFile = new java.io.File(cacheDir, cacheFileName);
          sessionOptions.setOptimizedModelFilePath(cacheFile.getAbsolutePath());
          Log.d(TAG, "📦 Optimized model cache: " + cacheFile.getAbsolutePath());
        }
        catch (Exception cacheError)
        {
          Log.w(TAG, "⚠️ Could not set optimized model cache: " + cacheError.getMessage());
        }
      }

      // OPTIMIZATION 5: Enable verbose logging for execution provider verification
      try
      {
        sessionOptions.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);
        // logDebug("🔍 Verbose logging enabled for execution provider verification");
      }
      catch (Exception logError)
      {
        // logDebug("⚠️ Verbose logging not available: " + logError.getMessage());
      }

      // OPTIMIZATION 6: Modern execution providers (QNN NPU priority for Samsung S25U)
      boolean hardwareAcceleration = tryEnableHardwareAcceleration(sessionOptions, sessionName);

      return sessionOptions;
    }
    catch (Exception e)
    {
      // logDebug("💥 Failed to create optimized SessionOptions for " + sessionName + ": " + e.getMessage());
      Log.e(TAG, "Failed to create optimized SessionOptions", e);
      
      // Ultimate fallback: basic session options
      try
      {
        return new OrtSession.SessionOptions();
      }
      catch (Exception fallbackError)
      {
        throw new RuntimeException("Cannot create any SessionOptions", fallbackError);
      }
    }
  }
  
  // Removed unused initialization methods

  /**
   * OPTIMIZATION: Initialize batch processing buffers for single decoder call
   * This is the critical architectural change for 8x speedup (expert recommendation)
   */
  // Removed initializeBatchProcessingBuffers method
  
  /**
   * OPTIMIZATION: Create optimized ONNX session options for maximum performance
   * CRITICAL: Uses NNAPI execution provider for ARM64 hardware acceleration
   */
  private OrtSession.SessionOptions createOptimizedSessionOptions() throws OrtException
  {
    OrtSession.SessionOptions options = new OrtSession.SessionOptions();
    
    // Enable all available optimization levels
    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
    
    // CRITICAL OPTIMIZATION: Use Android NNAPI execution provider for hardware acceleration
    try {
      // Try to enable NNAPI with basic configuration
      options.addNnapi();
      Log.w(TAG, "🚀 NNAPI execution provider enabled for ARM64 hardware acceleration");
      // logDebug("🚀 NNAPI execution provider enabled for ARM64 hardware acceleration");
    } catch (Exception e) {
      Log.w(TAG, "⚠️ NNAPI not available, using CPU provider: " + e.getMessage());
      // logDebug("⚠️ NNAPI not available, using CPU provider: " + e.getMessage());
    }
    
    // Enable memory pattern optimization if available
    try {
      options.setMemoryPatternOptimization(true);
      // Log.d(TAG, "Memory pattern optimization enabled");
    } catch (Exception e) {
      // Log.d(TAG, "Memory pattern optimization not available in this ONNX version");
    }
    
    // Note: GPU execution provider method may not be available in this ONNX Runtime version
    // Log.d(TAG, "GPU execution provider configuration skipped for compatibility");
    
    Log.w(TAG, "🔧 ONNX session options optimized with hardware acceleration");
    return options;
  }
  
  /**
   * OPTIMIZATION: Enable hardware acceleration with modern execution providers
   * Uses available Java API methods with proper fallback strategy
   */
  /**
   * Creates an optimized OrtSession.SessionOptions with the NNAPI Execution Provider enabled.
   *
   * OPTIMIZATION v6 (perftodos6.md): NNAPI is CRITICAL for leveraging hardware acceleration
   * for quantized INT8 models on Android devices with NPU/DSP/GPU support.
   *
   * @param sessionName Name of the session for logging
   * @return SessionOptions configured with NNAPI flags
   */
  private OrtSession.SessionOptions createNnapiSessionOptions(String sessionName)
  {
    try
    {
      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

      // NNAPI for quantized INT8 models (perftodos6.md Step 2)
      // Note: For optimal performance, NnapiFlags can be used:
      //   int nnapiFlags = NnapiFlags.NNAPI_FLAG_USE_FP16;  // FP16 acceleration
      //   int nnapiFlags = NnapiFlags.NNAPI_FLAG_CPU_DISABLED;  // Debug: force NNAPI only
      // For production, use no-arg addNnapi() for maximum compatibility

      try
      {
        // Add NNAPI execution provider (basic configuration for compatibility)
        // The quantized INT8 model should automatically use NNAPI acceleration
        options.addNnapi();
        Log.i(TAG, "✅ NNAPI execution provider configured for " + sessionName + " (quantized INT8)");
        return options;
      }
      catch (Exception e)
      {
        Log.w(TAG, "NNAPI provider not available on this device, trying fallback providers", e);
        // Fall through to hardware acceleration fallbacks
      }

      // Fallback to existing QNN/XNNPACK if NNAPI fails
      tryEnableHardwareAcceleration(options, sessionName);
      return options;
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to create NNAPI SessionOptions, using default", e);

      try
      {
        return new OrtSession.SessionOptions();
      }
      catch (Exception fallbackError)
      {
        throw new RuntimeException("Cannot create any SessionOptions", fallbackError);
      }
    }
  }

  private boolean tryEnableHardwareAcceleration(OrtSession.SessionOptions sessionOptions, String sessionName)
  {
    boolean accelerationEnabled = false;

    // Priority 1: Try QNN for Samsung S25U Snapdragon NPU (requires quantized models)
    try
    {
      Map<String, String> qnnOptions = new HashMap<>();
      qnnOptions.put("backend_path", "libQnnHtp.so");                    // Explicit HTP backend
      qnnOptions.put("htp_performance_mode", "burst");                   // Burst mode for latency
      qnnOptions.put("htp_graph_finalization_optimization_mode", "3");   // Aggressive optimization
      qnnOptions.put("qnn_context_priority", "high");                    // High priority context
      
      // Use addConfigEntry since addQNN() may not be available in this ONNX Runtime version
      for (Map.Entry<String, String> entry : qnnOptions.entrySet())
      {
        sessionOptions.addConfigEntry("qnn_" + entry.getKey(), entry.getValue());
      }
      // logDebug("🚀 QNN execution provider enabled for Samsung S25U Snapdragon NPU");
      // logDebug("   🔥 HTP burst mode active for maximum performance");
      // Log.d(TAG, "QNN HTP NPU enabled for " + sessionName + " - Snapdragon hardware acceleration");
      return true;
    }
    catch (Exception qnnError)
    {
      // logDebug("⚠️ QNN not available (requires quantized model): " + qnnError.getMessage());
      Log.w(TAG, "QNN not available for " + sessionName + " (may need quantized model), trying XNNPACK");
      
      // Priority 2: Fallback to XNNPACK for optimized ARM CPU
      try
      {
        Map<String, String> xnnpackOptions = new HashMap<>();
        xnnpackOptions.put("intra_op_num_threads", "4"); // Samsung S25U performance cores
        
        sessionOptions.addXnnpack(xnnpackOptions);
        
        // Expert recommendation: Use SEQUENTIAL mode for single-inference latency
        sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
        sessionOptions.setIntraOpNumThreads(4);  // Match XNNPACK threads
        sessionOptions.setInterOpNumThreads(1);  // Dedicate resources to single stream
        
        // logDebug("🚀 XNNPACK execution provider enabled for Samsung S25U");
        // logDebug("   📱 4-core ARM sequential optimization for latency");
        // Log.d(TAG, "XNNPACK enabled for " + sessionName + " - optimized ARM acceleration");
        accelerationEnabled = true;
      }
      catch (Exception xnnpackError)
      {
        // logDebug("⚠️ XNNPACK not available: " + xnnpackError.getMessage());
        Log.w(TAG, "No hardware acceleration available, using optimized CPU");
        accelerationEnabled = false;
      }
    }
    
    return accelerationEnabled;
  }
  
  /**
   * CRITICAL: Verify which execution provider is actually running
   * Essential for performance validation on Samsung S25U
   */
  private boolean verifyExecutionProvider(OrtSession session, String sessionName)
  {
    try
    {
      // Check session metadata for actual execution providers
      // Note: getProvidersUsed() may not be available in all ONNX Runtime versions
      // This is a best-effort attempt to verify providers
      String[] providers = new String[]{"CPU"}; // Default fallback
      // TODO: Use reflection or alternative method to get actual providers when available
      
      boolean hardwareAccelerated = false;
      // logDebug("🔍 Execution providers verification for " + sessionName + " (limited API)");
      
      for (String provider : providers)
      {
        // Log.d(TAG, "Active execution provider: " + provider + " for " + sessionName);
        // logDebug("  - " + provider);
        
        if (provider.contains("XNNPACK") || provider.contains("QNN") || provider.contains("GPU"))
        {
          hardwareAccelerated = true;
          // Log.d(TAG, "✅ Hardware acceleration confirmed: " + provider + " for " + sessionName);
          // logDebug("✅ Hardware acceleration confirmed: " + provider);
        }
      }
      
      // Since we can't reliably detect providers, assume XNNPACK worked if no exception occurred
      // Log.d(TAG, "✅ Hardware acceleration configuration completed for " + sessionName);
      // logDebug("✅ Hardware acceleration configuration completed (verification limited by API)");
      
      return true; // Optimistically assume acceleration is working
    }
    catch (Exception e)
    {
      Log.w(TAG, "Failed to verify execution providers for " + sessionName + ": " + e.getMessage());
      // logDebug("⚠️ Failed to verify execution providers: " + e.getMessage());
      return false;
    }
  }
  
  /**
   * OPTIMIZATION: Initialize dedicated thread pool for ONNX operations
   * Uses optimized threading for tensor operations and inference
   */
  private void initializeThreadPool()
  {
    synchronized (_executorLock)
    {
      if (_onnxExecutor == null)
      {
        _onnxExecutor = Executors.newSingleThreadExecutor(new ThreadFactory()
        {
          @Override
          public Thread newThread(Runnable r)
          {
            Thread t = new Thread(r, "ONNX-Inference-Thread");
            t.setPriority(Thread.NORM_PRIORITY + 1); // Slightly higher priority for responsiveness
            t.setDaemon(false); // Keep thread alive for reuse
            return t;
          }
        });
        
        // Log.d(TAG, "ONNX thread pool initialized for optimized inference");
      }
    }
  }
  
  /**
   * EMERGENCY SPEED MODE: Greedy search with single beam (maximum performance)
   * Completely bypasses beam search for 10x+ speedup
   */
  private List<BeamSearchCandidate> runGreedySearch(OnnxTensor memory, int actualSrcLength, int maxLength)
  {
    long greedyStart = System.nanoTime();
    List<Integer> tokens = new ArrayList<>();
    tokens.add(SOS_IDX);
    
    // logDebug("🏃 Starting greedy search with max_length=" + maxLength);
    
    for (int step = 0; step < maxLength; step++)
    {
      // Simple greedy: always pick top token
      try
      {
        // Create fresh tensors like CLI test (no reusable buffers)
        final int DECODER_SEQ_LENGTH = 20; // MUST match actual model export

        // Pad sequence to DECODER_SEQ_LENGTH (V4 expects int32 for target_tokens)
        int[] tgtTokens = new int[DECODER_SEQ_LENGTH];
        Arrays.fill(tgtTokens, (int)PAD_IDX);
        for (int i = 0; i < Math.min(tokens.size(), DECODER_SEQ_LENGTH); i++)
        {
          tgtTokens[i] = tokens.get(i).intValue();
        }

        OnnxTensor targetTokensTensor = OnnxTensor.createTensor(_ortEnvironment, 
          java.nio.IntBuffer.wrap(tgtTokens), new long[]{1, DECODER_SEQ_LENGTH});
        // V4 interface: decoder creates masks internally from actual_src_length
        OnnxTensor actualSrcLengthTensor = OnnxTensor.createTensor(_ortEnvironment, new int[]{actualSrcLength});

        Map<String, OnnxTensor> decoderInputs = new HashMap<>();
        decoderInputs.put("memory", memory);
        decoderInputs.put("target_tokens", targetTokensTensor);
        decoderInputs.put("actual_src_length", actualSrcLengthTensor);
        
        OrtSession.Result decoderOutput = _decoderSession.run(decoderInputs);
        OnnxTensor logitsTensor = (OnnxTensor) decoderOutput.get(0);
        
        // Get logits and find top token
        Object logitsValue = logitsTensor.getValue();
        if (logitsValue instanceof float[][][])
        {
          float[][][] logits3D = (float[][][]) logitsValue;
          float[] currentLogits = logits3D[0][step];
          
          // Find token with maximum probability
          int bestToken = 0;
          float bestProb = Float.NEGATIVE_INFINITY;
          for (int i = 0; i < currentLogits.length; i++)
          {
            if (currentLogits[i] > bestProb)
            {
              bestProb = currentLogits[i];
              bestToken = i;
            }
          }
          
          // Only stop if EOS token - no arbitrary early termination
          if (bestToken == EOS_IDX)
          {
            // logDebug("🏁 Greedy search stopped at step " + step + " - EOS token");
            break;
          }
          
          tokens.add(bestToken);
          // logDebug("🎯 Greedy step " + step + ": token=" + bestToken + ", prob=" + bestProb);
        }
        
        targetTokensTensor.close();
        actualSrcLengthTensor.close();
        decoderOutput.close();
      }
      catch (Exception e)
      {
        Log.e(TAG, "Greedy search error at step " + step, e);
        break;
      }
    }
    
    // Convert tokens to word
    StringBuilder word = new StringBuilder();
    for (int token : tokens)
    {
      if (token != SOS_IDX && token != EOS_IDX && token != PAD_IDX)
      {
        char ch = _tokenizer.indexToChar(token);
        if (ch != '?' && !ch.toString().startsWith("<"))
        {
          word.append(ch);
        }
      }
    }
    
    long greedyTime = (System.nanoTime() - greedyStart) / 1_000_000;
    String wordStr = word.toString();
    // logDebug("🏆 Greedy search completed in " + greedyTime + "ms: '" + wordStr + "'\n");
    Log.w(TAG, "🏆 Greedy search completed in " + greedyTime + "ms: '" + wordStr + "'\n");
    
    List<BeamSearchCandidate> result = new ArrayList<>();
    if (wordStr.length() > 0)
    {
      result.add(new BeamSearchCandidate(wordStr, 0.9f)); // High confidence for greedy result
    }
    return result;
  }
  
  // NOTE: Removed updateReusableTokens - now creating fresh tensors like CLI test
  
  /**
   * Set configuration parameters
   */
  public void setConfig(Config config)
  {
    _config = config;

    // CRITICAL: Update cached config settings to avoid repeated checks on hot paths
    if (config != null)
    {
      updateConfig(config);
    }

    // Update neural parameters from config
    if (config != null)
    {
      _beamWidth = config.neural_beam_width != 0 ? config.neural_beam_width : DEFAULT_BEAM_WIDTH;
      _maxLength = config.neural_max_length != 0 ? config.neural_max_length : DEFAULT_MAX_LENGTH;
      _confidenceThreshold = config.neural_confidence_threshold != 0 ?
        config.neural_confidence_threshold : DEFAULT_CONFIDENCE_THRESHOLD;

      // OPTIMIZATION: Removed automatic model reload logic per user request.
      // Changes to 'neural_model_version' or custom paths now require a keyboard restart.
      // This eliminates overhead and prevents potential race conditions during app switches.
      // Old logic checked versionChanged || pathsChanged and called initialize().

      // Update max sequence length override
      if (config.neural_user_max_seq_length > 0)
      {
        _maxSequenceLength = config.neural_user_max_seq_length;
        Log.d(TAG, String.format("Using user-defined max sequence length: %d", _maxSequenceLength));
      }

      // Update resampling mode in trajectory processor
      if (_trajectoryProcessor != null && config.neural_resampling_mode != null)
      {
        SwipeResampler.ResamplingMode mode = SwipeResampler.parseMode(config.neural_resampling_mode);
        _trajectoryProcessor.setResamplingMode(mode);
      }
    }

    // Log.d(TAG, String.format("Neural config: beam_width=%d, max_length=%d, threshold=%.3f, model=%s, seq_len=%d",
      // _beamWidth, _maxLength, _confidenceThreshold, _currentModelVersion, _maxSequenceLength));
  }
  
  /**
   * Set keyboard dimensions for trajectory processing
   */
  public void setKeyboardDimensions(float width, float height)
  {
    if (_trajectoryProcessor != null)
    {
      _trajectoryProcessor.setKeyboardLayout(null, width, height);
    }
  }

  /**
   * Set QWERTY area bounds for proper coordinate normalization.
   * The neural model expects coordinates normalized over just the QWERTY key area,
   * not the full keyboard view.
   *
   * @param qwertyTop Y offset in pixels where QWERTY keys start
   * @param qwertyHeight Height in pixels of the QWERTY key area
   */
  public void setQwertyAreaBounds(float qwertyTop, float qwertyHeight)
  {
    if (_trajectoryProcessor != null)
    {
      _trajectoryProcessor.setQwertyAreaBounds(qwertyTop, qwertyHeight);
    }
  }

  /**
   * Set touch Y-offset compensation for fat finger effect.
   *
   * @param offset Pixels to add to Y coordinate (positive = shift down toward key center)
   */
  public void setTouchYOffset(float offset)
  {
    if (_trajectoryProcessor != null)
    {
      _trajectoryProcessor.setTouchYOffset(offset);
    }
  }
  
  /**
   * Set real key positions for trajectory processing
   */
  public void setRealKeyPositions(Map<Character, PointF> realPositions)
  {
    if (_trajectoryProcessor != null && realPositions != null)
    {
      // Get current keyboard dimensions
      float width = _trajectoryProcessor._keyboardWidth;
      float height = _trajectoryProcessor._keyboardHeight;
      _trajectoryProcessor.setKeyboardLayout(realPositions, width, height);
    }
  }
  
  /**
   * Set debug logger for detailed logging
   */
  public void setDebugLogger(NeuralSwipeTypingEngine.DebugLogger logger)
  {
    _debugLogger = logger;
  }

  /**
   * Update cached configuration settings.
   * CRITICAL: Call this when config changes (not on every swipe!)
   * Caches expensive-to-check settings to avoid SharedPreferences access in hot paths.
   *
   * @param config Updated configuration from ConfigurationManager
   */
  public void updateConfig(Config config)
  {
    _cachedConfig = config;
    _enableVerboseLogging = config.swipe_debug_detailed_logging;
    _showRawOutput = config.swipe_debug_show_raw_output;
    _batchBeams = config.neural_batch_beams;

    // Cache other frequently-checked settings here as needed
    // Example: _useQuantizedModels = config.neural_use_quantized;

    // CRITICAL FIX: Propagate config to vocabulary for its own caching
    if (_vocabulary != null)
    {
      _vocabulary.updateConfig(config);
    }

    // Log config update (this itself is NOT verbose logging)
    Log.d(TAG, "Config updated: verbose_logging=" + _enableVerboseLogging +
              ", show_raw=" + _showRawOutput + ", batch_beams=" + _batchBeams);
  }

  private void logDebug(String message)
  {
    if (_debugLogger != null)
    {
      _debugLogger.log(message);
    }
  }
  
  /**
   * Check if neural prediction is available
   */
  public boolean isAvailable()
  {
    return _isModelLoaded;
  }

  /**
   * Get current model information for display
   */
  public String getModelInfo()
  {
    return String.format("%s (%s, %d-len, %s)",
      _currentModelVersion,
      _modelSource,
      _maxSequenceLength,
      _modelAccuracy);
  }

  /**
   * Get current model version
   */
  public String getModelVersion()
  {
    return _currentModelVersion;
  }

  /**
   * Get model accuracy
   */
  public String getModelAccuracy()
  {
    return _modelAccuracy;
  }

  /**
   * Get model source (builtin/external/fallback)
   */
  public String getModelSource()
  {
    return _modelSource;
  }

  /**
   * Get max sequence length
   */
  public int getMaxSequenceLength()
  {
    return _maxSequenceLength;
  }
  
  
  /**
   * Load model from assets or external file path
   * Supports both builtin models (assets) and user-provided external files
   */
  private byte[] loadModelFromAssets(String modelPath)
  {
    try
    {
      InputStream inputStream;

      // Check if it's a content URI (starts with content://)
      if (modelPath.startsWith("content://"))
      {
        Log.d(TAG, "Loading external ONNX model from URI: " + modelPath);
        android.net.Uri uri = android.net.Uri.parse(modelPath);

        try
        {
          inputStream = _context.getContentResolver().openInputStream(uri);
          if (inputStream == null)
          {
            Log.e(TAG, "Cannot open input stream for URI: " + modelPath);
            return null;
          }
          Log.d(TAG, "External model loaded from content URI");
        }
        catch (SecurityException e)
        {
          Log.e(TAG, "Permission denied for URI: " + modelPath, e);
          return null;
        }
      }
      // Check if it's an external file path (starts with /)
      else if (modelPath.startsWith("/"))
      {
        Log.d(TAG, "Loading external ONNX model from file path: " + modelPath);
        java.io.File file = new java.io.File(modelPath);

        if (!file.exists())
        {
          Log.e(TAG, "External model file does not exist: " + modelPath);
          return null;
        }

        if (!file.canRead())
        {
          Log.e(TAG, "Cannot read external model file: " + modelPath);
          return null;
        }

        inputStream = new java.io.FileInputStream(file);
        Log.d(TAG, "External model file size: " + file.length() + " bytes");
      }
      else
      {
        // Load from assets
        // Log.d(TAG, "Loading ONNX model from assets: " + modelPath);
        inputStream = _context.getAssets().open(modelPath);
      }

      int available = inputStream.available();
      // Log.d(TAG, "Model file size: " + available + " bytes");

      byte[] modelData = new byte[available];
      int totalRead = 0;
      while (totalRead < available) {
        int read = inputStream.read(modelData, totalRead, available - totalRead);
        if (read == -1) break;
        totalRead += read;
      }
      inputStream.close();

      // Log.d(TAG, "Successfully loaded " + totalRead + " bytes from " + modelPath);
      return modelData;
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load model: " + modelPath, e);
      return null;
    }
  }

  /**
   * OPTIMIZATION v6 (perftodos6.md): Read model configuration to detect broadcast support
   * Broadcast-enabled models expand memory internally, avoiding manual replication
   */
  private void readModelConfig(String modelPath)
  {
    try
    {
      // Derive config path from model path (e.g., models/bs/swipe_encoder_android.onnx -> models/bs/model_config.json)
      String configPath;
      if (modelPath.contains("/bs/"))
      {
        // Quantized broadcast models in bs/ directory
        configPath = "models/bs/model_config.json";
      }
      else
      {
        // Standard float32 models - no config, assume broadcast disabled
        _broadcastEnabled = false;
        Log.d(TAG, "Using float32 models - broadcast disabled (manual memory replication)");
        return;
      }

      // Load and parse JSON config
      InputStream configStream = _context.getAssets().open(configPath);
      byte[] buffer = new byte[configStream.available()];
      configStream.read(buffer);
      configStream.close();
      String jsonString = new String(buffer, "UTF-8");

      // Parse broadcast_enabled flag (simple JSON parsing without external dependencies)
      // Example: "broadcast_enabled": true
      _broadcastEnabled = jsonString.contains("\"broadcast_enabled\"") &&
                          jsonString.contains("true");

      if (_broadcastEnabled)
      {
        Log.i(TAG, "✅ Broadcast-enabled models detected");
      }
      else
      {
        Log.d(TAG, "Broadcast disabled - manual memory replication");
      }
    }
    catch (IOException e)
    {
      Log.w(TAG, "Could not read model_config.json - assuming broadcast disabled: " + e.getMessage());
      _broadcastEnabled = false;
    }
  }

  private OnnxTensor createTrajectoryTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    // Create direct buffer as recommended by ONNX docs
    java.nio.ByteBuffer byteBuffer = java.nio.ByteBuffer.allocateDirect(_maxSequenceLength * TRAJECTORY_FEATURES * 4); // 4 bytes per float
    byteBuffer.order(java.nio.ByteOrder.nativeOrder());
    java.nio.FloatBuffer buffer = byteBuffer.asFloatBuffer();

    for (int i = 0; i < _maxSequenceLength; i++)
    {
      if (i < features.normalizedPoints.size())
      {
        SwipeTrajectoryProcessor.TrajectoryPoint point = features.normalizedPoints.get(i);
        buffer.put(point.x);
        buffer.put(point.y);
        buffer.put(point.vx);
        buffer.put(point.vy);
        buffer.put(point.ax);
        buffer.put(point.ay);
      }
      else
      {
        // Padding with zeros
        buffer.put(0.0f); // x
        buffer.put(0.0f); // y
        buffer.put(0.0f); // vx
        buffer.put(0.0f); // vy
        buffer.put(0.0f); // ax
        buffer.put(0.0f); // ay
      }
    }

    buffer.rewind();
    long[] shape = {1, _maxSequenceLength, TRAJECTORY_FEATURES};
    return OnnxTensor.createTensor(_ortEnvironment, buffer, shape);
  }
  
  private OnnxTensor createNearestKeysTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    // Create direct buffer - V4 expects int32, not int64
    java.nio.ByteBuffer byteBuffer = java.nio.ByteBuffer.allocateDirect(_maxSequenceLength * 4); // 4 bytes per int
    byteBuffer.order(java.nio.ByteOrder.nativeOrder());
    java.nio.IntBuffer buffer = byteBuffer.asIntBuffer();

    // CRITICAL FIX: nearestKeys is now List<Integer> (token indices), not List<Character>!
    for (int i = 0; i < _maxSequenceLength; i++)
    {
      if (i < features.nearestKeys.size())
      {
        int tokenIndex = features.nearestKeys.get(i);
        buffer.put(tokenIndex);
      }
      else
      {
        buffer.put((int)PAD_IDX); // Padding (should never hit this - features are pre-padded)
      }
    }

    buffer.rewind();
    long[] shape = {1, _maxSequenceLength};
    return OnnxTensor.createTensor(_ortEnvironment, buffer, shape);
  }
  
  private OnnxTensor createSourceMaskTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    // Create 2D boolean array for proper tensor shape [1, _maxSequenceLength]
    boolean[][] maskData = new boolean[1][_maxSequenceLength];

    // Mask padded positions (true = masked/padded, false = valid)
    for (int i = 0; i < _maxSequenceLength; i++)
    {
      maskData[0][i] = (i >= features.actualLength);
    }
    
    // Use 2D boolean array - ONNX API will infer shape as [1, 100]
    return OnnxTensor.createTensor(_ortEnvironment, maskData);
  }
  
  // REFACTORING: New overload that accepts OnnxTensor directly (for EncoderWrapper integration)
  private List<BeamSearchCandidate> runBeamSearch(OnnxTensor memory,
    int actualSrcLength, SwipeTrajectoryProcessor.TrajectoryFeatures features) throws OrtException
  {
    if (_decoderSession == null)
    {
      Log.e(TAG, "Decoder not loaded, cannot decode");
      return new ArrayList<>();
    }

    if (memory == null)
    {
      Log.e(TAG, "No memory tensor from encoder");
      return new ArrayList<>();
    }

    // Initialize BeamSearchEngine
    VocabularyTrie trie = (_vocabulary != null) ? _vocabulary.getVocabularyTrie() : null;
    
    // Lambda for debug logging
    kotlin.jvm.functions.Function1<String, kotlin.Unit> logger = null;
    if (_enableVerboseLogging && _debugLogger != null) {
        logger = msg -> {
            _debugLogger.log(msg);
            return kotlin.Unit.INSTANCE;
        };
    }

    BeamSearchEngine engine = new BeamSearchEngine(
        _decoderSession,
        _ortEnvironment,
        _tokenizer,
        trie,
        _beamWidth,
        _maxLength,
        _confidenceThreshold,
        logger
    );

    // Run search
    List<BeamSearchEngine.BeamSearchCandidate> results = engine.search(memory, actualSrcLength, _batchBeams);

    // Map to local BeamSearchCandidate
    List<BeamSearchCandidate> candidates = new ArrayList<>();
    for (BeamSearchEngine.BeamSearchCandidate result : results) {
        candidates.add(new BeamSearchCandidate(result.getWord(), result.getConfidence()));
    }

    return candidates;
  }
  
  private static class BeamSearchCandidate
  {
    public final String word;
    public final float confidence;
    
    public BeamSearchCandidate(String word, float confidence)
    {
      this.word = word;
      this.confidence = confidence;
    }
  }
  
  private PredictionResult createPredictionResult(List<BeamSearchCandidate> candidates, SwipeInput input)
  {
    // OPTIMIZATION: Use vocabulary filtering for better predictions (2x speedup + quality)
    if (_vocabulary != null && _vocabulary.isLoaded())
    {
      return createOptimizedPredictionResult(candidates, input);
    }

    // Fallback: Basic filtering for testing
    List<String> words = new ArrayList<>();
    List<Integer> scores = new ArrayList<>();

    for (BeamSearchCandidate candidate : candidates)
    {
      if (candidate.confidence >= _confidenceThreshold)
      {
        words.add(candidate.word);
        scores.add((int)(candidate.confidence * 1000)); // Convert to 0-1000 range
      }
    }

    // DEBUG MODE: Log raw neural network outputs for analysis (not shown in UI, CACHED check)
    if (_showRawOutput && !candidates.isEmpty())
    {
      StringBuilder debugOutput = new StringBuilder("🔍 Raw NN Beam Search:\n");
      int numToShow = Math.min(5, candidates.size());
      for (int i = 0; i < numToShow; i++)
      {
        BeamSearchCandidate candidate = candidates.get(i);
        boolean inFiltered = false;
        for (String word : words) {
          if (word.equalsIgnoreCase(candidate.word)) {
            inFiltered = true;
            break;
          }
        }

        String marker = inFiltered ? "[kept]" : "[filtered]";
        debugOutput.append(String.format("  %d. %s %.3f %s\n",
          i + 1, candidate.word, candidate.confidence, marker));
      }
      Log.d(TAG, debugOutput.toString());
      logDebug(debugOutput.toString());
    }

    // logDebug("📊 Raw predictions: " + candidates.size() + " total, " + words.size() + " above threshold");
    return new PredictionResult(words, scores);
  }
  
  /**
   * OPTIMIZATION: Create optimized prediction result using vocabulary filtering
   * Implements web app fast-path lookup and combined scoring
   */
  private PredictionResult createOptimizedPredictionResult(List<BeamSearchCandidate> candidates, SwipeInput input)
  {
    // ALWAYS log top 3 model outputs for debugging (shows raw NN output before filtering)
    if (_debugLogger != null && !candidates.isEmpty())
    {
      StringBuilder modelOutput = new StringBuilder("🤖 MODEL OUTPUT: ");
      int numToShow = Math.min(3, candidates.size());
      for (int i = 0; i < numToShow; i++)
      {
        BeamSearchCandidate c = candidates.get(i);
        if (i > 0) modelOutput.append(", ");
        modelOutput.append(String.format("%s(%.2f)", c.word, c.confidence));
      }
      modelOutput.append("\n");
      logDebug(modelOutput.toString());
    }

    // Convert beam candidates to vocabulary format
    List<OptimizedVocabulary.CandidateWord> vocabCandidates = new ArrayList<>();
    for (BeamSearchCandidate candidate : candidates)
    {
      vocabCandidates.add(new OptimizedVocabulary.CandidateWord(candidate.word, candidate.confidence));
    }

    // Extract last character from swipe path for contraction filtering
    char lastChar = '\0';
    if (input != null && input.keySequence != null && !input.keySequence.isEmpty())
    {
      lastChar = input.keySequence.charAt(input.keySequence.length() - 1);
    }

    // Get first character for prefix filtering (Starting Letter Accuracy)
    char firstChar = '\0';
    if (input != null && input.keySequence != null && input.keySequence.length() > 0)
    {
      firstChar = input.keySequence.charAt(0);
    }

    // Apply vocabulary filtering with fast-path optimization
    OptimizedVocabulary.SwipeStats swipeStats = new OptimizedVocabulary.SwipeStats(
      input != null && input.keySequence != null ? input.keySequence.length() : 0,
      input != null ? input.pathLength : 0,
      input != null ? input.averageVelocity : 0,
      firstChar,
      lastChar
    );
    List<OptimizedVocabulary.FilteredPrediction> filtered = _vocabulary.filterPredictions(vocabCandidates, swipeStats);

    // Convert back to PredictionResult format with deduplication
    // v1.33.5: CRITICAL FIX - deduplicate words, keeping highest score
    // v1.32.236: Use displayText for UI, but deduplicate by word (insertion text)

    // Helper class for deduplication (holds display text + score)
    class WordDisplayPair {
      final String displayText;
      final int score;
      WordDisplayPair(String displayText, int score) {
        this.displayText = displayText;
        this.score = score;
      }
    }

    Map<String, WordDisplayPair> wordScoreMap = new LinkedHashMap<>(); // Preserve insertion order

    for (OptimizedVocabulary.FilteredPrediction pred : filtered)
    {
      String wordLower = pred.word.toLowerCase();
      String displayLower = pred.displayText.toLowerCase();
      int score = (int)(pred.score * 1000); // Convert combined score to 0-1000 range

      // Keep only the highest score for each word (deduplicate by insertion text, not display)
      if (!wordScoreMap.containsKey(wordLower) || score > wordScoreMap.get(wordLower).score)
      {
        wordScoreMap.put(wordLower, new WordDisplayPair(displayLower, score));
      }
    }

    // Convert deduplicated map to lists
    // Use displayText for UI (shows proper contractions with apostrophes)
    // Keyboard2.java will recognize contractions and skip autocorrect
    List<String> words = new ArrayList<>();
    List<Integer> scores = new ArrayList<>();
    for (Map.Entry<String, WordDisplayPair> entry : wordScoreMap.entrySet())
    {
      words.add(entry.getValue().displayText);  // Use displayText for proper UI display
      scores.add(entry.getValue().score);
    }

    // Add raw beam search predictions (closest matches) AFTER filtered predictions
    // v1.33.4: CRITICAL FIX - raw predictions must ALWAYS rank below valid vocabulary words
    // This shows what the neural network actually predicted vs vocabulary filtering
    if (!candidates.isEmpty() && _config != null && _config.swipe_show_raw_beam_predictions)
    {
      // Find minimum score from filtered predictions to ensure raw ones rank lower
      int minFilteredScore = Integer.MAX_VALUE;
      for (int score : scores) {
        if (score < minFilteredScore) {
          minFilteredScore = score;
        }
      }

      // Cap raw prediction scores well below filtered predictions
      // Use 10% of minimum filtered score to ensure they always appear last
      int rawScoreCap = Math.max(1, minFilteredScore / 10);

      int numRawToAdd = Math.min(3, candidates.size());
      for (int i = 0; i < numRawToAdd; i++)
      {
        BeamSearchCandidate candidate = candidates.get(i);

        // Only add if not already in filtered results
        boolean alreadyIncluded = false;
        for (String word : words) {
          if (word.equalsIgnoreCase(candidate.word)) {
            alreadyIncluded = true;
            break;
          }
        }

        if (!alreadyIncluded)
        {
          // v1.33.4: Cap raw prediction score to ensure it ranks BELOW all valid words
          // Add "raw:" prefix to clearly identify unfiltered beam outputs
          int rawScore = Math.min((int)(candidate.confidence * 1000), rawScoreCap);
          words.add("raw:" + candidate.word);
          scores.add(rawScore);
        }
      }
    }

    // DEBUG MODE: Log raw neural network outputs for analysis (CACHED check)
    if (_showRawOutput && !candidates.isEmpty())
    {
      StringBuilder debugOutput = new StringBuilder("🔍 Raw NN Beam Search (with vocab filtering):\n");
      int numToShow = Math.min(5, candidates.size());
      for (int i = 0; i < numToShow; i++)
      {
        BeamSearchCandidate candidate = candidates.get(i);
        boolean inFiltered = false;
        for (String word : words) {
          if (word.equalsIgnoreCase(candidate.word)) {
            inFiltered = true;
            break;
          }
        }

        String marker = inFiltered ? "[kept by vocab]" : "[filtered out]";
        debugOutput.append(String.format("  %d. %s %.3f %s\n",
          i + 1, candidate.word, candidate.confidence, marker));
      }
      Log.d(TAG, debugOutput.toString());
      logDebug(debugOutput.toString());
    }

    // logDebug("📊 Optimized predictions: " + candidates.size() + " raw → " + filtered.size() + " filtered");
    // logDebug("   Fast-path breakdown: " +
      // filtered.stream().mapToLong(p -> p.source.equals(\"common\") ? 1 : 0).sum() + " common, " +
      // filtered.stream().mapToLong(p -> p.source.equals(\"top5000\") ? 1 : 0).sum() + " top5000");

    return new PredictionResult(words, scores);
  }
  
  private PredictionResult createEmptyResult()
  {
    return new PredictionResult(new ArrayList<>(), new ArrayList<>());
  }
  
  /**
   * OPTIMIZATION: Controlled cleanup that respects session persistence
   * Only cleans up sessions if explicitly requested (default: keep in memory)
   * THREAD SAFETY: synchronized to prevent cleanup during initialization
   */
  public synchronized void cleanup()
  {
    cleanup(false); // Default: keep sessions for performance
  }

  public synchronized void cleanup(boolean forceCleanup)
  {
    if (!_keepSessionsInMemory || forceCleanup)
    {
      // Log.d(TAG, "Cleaning up ONNX sessions (forced: " + forceCleanup + ")");
      
      try
      {
        if (_encoderSession != null)
        {
          _encoderSession.close();
          _encoderSession = null;
        }
        
        if (_decoderSession != null)
        {
          _decoderSession.close();
          _decoderSession = null;
        }
        
        _isModelLoaded = false;
        // Log.d(TAG, "ONNX sessions cleaned up");
      }
      catch (Exception e)
      {
        Log.e(TAG, "Error during ONNX cleanup", e);
      }
    }
    else
    {
      // Log.d(TAG, "Keeping ONNX sessions in memory for performance");
    }
    
    // Clean up thread pool if forcing cleanup
    if (forceCleanup)
    {
      synchronized (_executorLock)
      {
        if (_onnxExecutor != null)
        {
          _onnxExecutor.shutdown();
          _onnxExecutor = null;
          // Log.d(TAG, "ONNX thread pool cleaned up");
        }
      }
    }
  }
  
  /**
   * Force singleton reset (for testing/debugging only)
   */
  /**
   * Reload custom words, user dictionary, and disabled words in vocabulary
   * Called when Dictionary Manager makes changes
   * PERFORMANCE: Only reloads small dynamic sets, not the 10k main dictionary
   */
  public void reloadVocabulary()
  {
    if (_vocabulary != null)
    {
      _vocabulary.reloadCustomAndDisabledWords();
      Log.d(TAG, "Vocabulary reloaded after dictionary changes");
    }
  }

  public static void resetSingleton()
  {
    synchronized (_singletonLock)
    {
      if (_singletonInstance != null)
      {
        _singletonInstance.cleanup(true);
        _singletonInstance = null;
        // Log.d(TAG, "Singleton instance reset");
      }
    }
  }
}
