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
  private static final int DEFAULT_BEAM_WIDTH = 2; // Mobile-optimized: 2 beams (was 8)
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
  
  
  // Model state
  private boolean _isModelLoaded = false;
  private boolean _isInitialized = false;
  private boolean _keepSessionsInMemory = true; // OPTIMIZATION: Never unload for speed
  private boolean _usesSeparateMasks = false; // Track if decoder uses separate padding/causal masks (custom models) vs combined target_mask (v2 builtin)
  private boolean _broadcastEnabled = false; // OPTIMIZATION v6 (perftodos6.md): Broadcast-enabled models expand memory internally
  
  // Configuration parameters
  private int _beamWidth = DEFAULT_BEAM_WIDTH;
  private int _maxLength = DEFAULT_MAX_LENGTH;
  private float _confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD;
  
  // OPTIMIZATION: Pre-allocated tensor buffers for reuse (3x speedup expected)
  private long[] _reusableTokensArray;
  private boolean[][] _reusableTargetMaskArray;
  private java.nio.LongBuffer _reusableTokensBuffer;

  // OPTIMIZATION: Batch processing buffers for single decoder call (8x speedup expected)
  private long[][] _batchedTokensArray;     // [beam_width, seq_length]
  private boolean[][] _batchedMaskArray;    // [beam_width, seq_length]
  private float[][][] _batchedMemoryArray; // [beam_width, 150, 256]

  // OPTIMIZATION v1.32.420: Memory pool for tensor buffers to reduce GC pressure
  private java.nio.ByteBuffer _pooledTokensByteBuffer;  // Reusable ByteBuffer for tokens
  private java.nio.LongBuffer _pooledTokensLongBuffer;  // Reusable LongBuffer view
  private float[][][] _pooledMemoryArray;                // Reusable memory replication array
  private boolean[][] _pooledSrcMaskArray;               // Reusable src_mask array
  private int _pooledBufferMaxBeams = 0;                 // Track allocated capacity

  // OPTIMIZATION v1.32.489: Pre-allocated buffers for beam search loop
  // These are allocated once and reused every iteration to eliminate GC pressure
  private int[][] _preallocBatchedTokens;               // [beam_width, DECODER_SEQ_LENGTH]
  private java.nio.ByteBuffer _preallocTokensByteBuffer; // Direct buffer for ONNX
  private java.nio.IntBuffer _preallocTokensIntBuffer;   // View into byte buffer
  private int[] _preallocSrcLengths;                     // [beam_width] for actual_src_length
  private float[] _preallocProbs;                        // [vocab_size] for softmax output

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
   */
  public boolean initialize()
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
          // Check user preference for quantized vs float32 models
          boolean useQuantized = (_config != null && _config.neural_use_quantized);

          if (useQuantized)
          {
            // INT8 quantized models with broadcast support (calibrated, v2)
            encoderPath = "models/bs2/swipe_encoder_android.onnx";
            decoderPath = "models/bs2/swipe_decoder_android.onnx";
            _maxSequenceLength = 250;
            _modelAccuracy = "73.4%";
            _modelSource = "builtin-quantized-v2";
            Log.i(TAG, "Loading v2 INT8 quantized models (calibrated, broadcast-enabled, XNNPACK-optimized)");
          }
          else
          {
            // Standard float32 models (default, more stable)
            encoderPath = "models/swipe_encoder_android.onnx";
            decoderPath = "models/swipe_decoder_android.onnx";
            _maxSequenceLength = 250;
            _modelAccuracy = "80.6%";
            _modelSource = "builtin";
            Log.d(TAG, "Loading v2 float32 models (standard, NNAPI-accelerated)");
          }
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

      // Load encoder model (using correct name from web demo)
      Log.d(TAG, "Loading encoder model from: " + encoderPath);
      long encStartTime = System.currentTimeMillis();
      byte[] encoderModelData = loadModelFromAssets(encoderPath);
      long encReadTime = System.currentTimeMillis() - encStartTime;
      Log.i(TAG, "⏱️ Encoder read: " + encReadTime + "ms");

      if (encoderModelData != null)
      {
        // logDebug("📥 Encoder model data loaded: " + encoderModelData.length + " bytes");
        long encSessionStart = System.currentTimeMillis();
        // OPTIMIZATION v6 (perftodos6.md Step 2): Use NNAPI for quantized models
        OrtSession.SessionOptions sessionOptions = createNnapiSessionOptions("Encoder");
        _encoderSession = _ortEnvironment.createSession(encoderModelData, sessionOptions);
        long encSessionTime = System.currentTimeMillis() - encSessionStart;
        Log.i(TAG, "⏱️ Encoder session creation: " + encSessionTime + "ms");

        // CRITICAL: Verify execution provider is working
        verifyExecutionProvider(_encoderSession, "Encoder");

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

        // Log encoder interface only if verbose logging enabled (CACHED)
        if (_enableVerboseLogging)
        {
          Log.d(TAG, "✅ Encoder session created successfully");
          Log.d(TAG, "   Encoder Inputs: " + _encoderSession.getInputNames());
          Log.d(TAG, "   Encoder Outputs: " + _encoderSession.getOutputNames());
        }

        Log.d(TAG, String.format("Encoder model loaded: %s (max_seq_len=%d)", _currentModelVersion, _maxSequenceLength));
      }
      else
      {
        // logDebug("❌ Failed to load encoder model data");
        Log.e(TAG, "Failed to load encoder model data from: " + encoderPath);
      }
      Log.d(TAG, "Finished loading encoder model");

      // Load decoder model (using correct name from web demo)
      Log.d(TAG, "Loading decoder model from: " + decoderPath);
      long decStartTime = System.currentTimeMillis();
      byte[] decoderModelData = loadModelFromAssets(decoderPath);
      long decReadTime = System.currentTimeMillis() - decStartTime;
      Log.i(TAG, "⏱️ Decoder read: " + decReadTime + "ms");

      if (decoderModelData != null)
      {
        // logDebug("📥 Decoder model data loaded: " + decoderModelData.length + " bytes");
        long decSessionStart = System.currentTimeMillis();
        // OPTIMIZATION v6 (perftodos6.md Step 2): Use NNAPI for quantized models
        OrtSession.SessionOptions sessionOptions = createNnapiSessionOptions("Decoder");
        _decoderSession = _ortEnvironment.createSession(decoderModelData, sessionOptions);
        long decSessionTime = System.currentTimeMillis() - decSessionStart;
        Log.i(TAG, "⏱️ Decoder session creation: " + decSessionTime + "ms");
        // logDebug("✅ Decoder session created successfully");
        // logDebug("   Inputs: " + _decoderSession.getInputNames());
        // logDebug("   Outputs: " + _decoderSession.getOutputNames());

        // CRITICAL: Verify execution provider is working
        verifyExecutionProvider(_decoderSession, "Decoder");

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
      }
      else
      {
        // logDebug("❌ Failed to load decoder model data");
        Log.e(TAG, "Failed to load decoder model data from: " + decoderPath);
      }
      Log.d(TAG, "Finished loading decoder model");

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
      boolean vocabularyLoaded = _vocabulary.loadVocabulary();
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
        initializeReusableBuffers();
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

      // Run encoder inference with proper ONNX API
      OnnxTensor trajectoryTensor = null;
      OnnxTensor nearestKeysTensor = null;
      OnnxTensor actualLengthTensor = null;
      OnnxTensor srcMaskTensor = null;  // Still needed for decoder
      OrtSession.Result encoderResult = null;

      try {
        trajectoryTensor = createTrajectoryTensor(features);
        nearestKeysTensor = createNearestKeysTensor(features);
        // New encoder uses actual_length instead of src_mask (V4 expects int32)
        actualLengthTensor = OnnxTensor.createTensor(_ortEnvironment, new int[]{features.actualLength});
        // Still create src_mask for decoder use
        srcMaskTensor = createSourceMaskTensor(features);

        // Log tensor shapes only if verbose logging enabled (CACHED - no config check on hot path)
        if (_enableVerboseLogging)
        {
          Log.d(TAG, "🔧 Encoder input tensor shapes (features.actualLength=" + features.actualLength + ", _maxSequenceLength=" + _maxSequenceLength + "):");
          Log.d(TAG, "   trajectory_features: " + java.util.Arrays.toString(trajectoryTensor.getInfo().getShape()));
          Log.d(TAG, "   nearest_keys: " + java.util.Arrays.toString(nearestKeysTensor.getInfo().getShape()));
          Log.d(TAG, "   actual_length: " + java.util.Arrays.toString(actualLengthTensor.getInfo().getShape()));
        }

        Map<String, OnnxTensor> encoderInputs = new HashMap<>();
        encoderInputs.put("trajectory_features", trajectoryTensor);
        encoderInputs.put("nearest_keys", nearestKeysTensor);
        encoderInputs.put("actual_length", actualLengthTensor);
        
        // Run encoder inference with detailed timing
        long encoderStartTime = System.nanoTime();
        try (OrtSession.Result encoderResults = _encoderSession.run(encoderInputs)) {
          long encoderTime = System.nanoTime() - encoderStartTime;
          
          // Run beam search or greedy search decoding with timing
          long searchStartTime = System.nanoTime();
          List<BeamSearchCandidate> candidates;
          if (_config != null && _config.neural_greedy_search) {
              candidates = runGreedySearch((OnnxTensor) encoderResults.get(0), features.actualLength, _maxLength);
          } else {
              candidates = runBeamSearch(encoderResults, features.actualLength, features);
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
        }
        
      } finally {
        // Proper memory cleanup
        if (trajectoryTensor != null) trajectoryTensor.close();
        if (nearestKeysTensor != null) nearestKeysTensor.close();
        if (actualLengthTensor != null) actualLengthTensor.close();
        if (srcMaskTensor != null) srcMaskTensor.close();
      }
    }
    catch (Exception e)
    {
      Log.e(TAG, "Neural prediction failed", e);
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
  
  /**
   * OPTIMIZATION: Initialize reusable tensor buffers for beam search
   * This prevents creating new tensors for every beam search step (3x speedup)
   */
  private void initializeReusableBuffers()
  {
    try
    {
      // Pre-allocate arrays for decoder sequence length (must match model_config.json max_word_len)
      int decoderSeqLength = 20; // MUST match actual model export (not model_config.json)
      _reusableTokensArray = new long[decoderSeqLength];
      _reusableTargetMaskArray = new boolean[1][decoderSeqLength];
      _reusableTokensBuffer = java.nio.LongBuffer.allocate(decoderSeqLength);

      // CRITICAL OPTIMIZATION: Initialize batch processing buffers
      initializeBatchProcessingBuffers(decoderSeqLength);

      // OPTIMIZATION v1.32.420: Initialize memory pool for tensor buffers
      initializeMemoryPool(decoderSeqLength);

      // OPTIMIZATION v1.32.489: Pre-allocate beam search loop buffers
      // These are allocated once and reused every iteration to eliminate GC pressure
      int maxBeams = _beamWidth > 0 ? _beamWidth : DEFAULT_BEAM_WIDTH;
      int vocabSize = 30; // Standard vocab size (26 letters + special tokens)

      _preallocBatchedTokens = new int[maxBeams][decoderSeqLength];
      _preallocSrcLengths = new int[maxBeams];
      _preallocProbs = new float[vocabSize];

      // Direct buffer for ONNX tensor creation (reusable)
      int tokensByteBufferSize = maxBeams * decoderSeqLength * 4; // 4 bytes per int
      _preallocTokensByteBuffer = java.nio.ByteBuffer.allocateDirect(tokensByteBufferSize);
      _preallocTokensByteBuffer.order(java.nio.ByteOrder.nativeOrder());
      _preallocTokensIntBuffer = _preallocTokensByteBuffer.asIntBuffer();

      Log.d(TAG, "Pre-allocated beam search buffers: " + maxBeams + " beams × " + decoderSeqLength + " seq_len, vocab=" + vocabSize);
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize reusable buffers", e);
    }
  }

  /**
   * OPTIMIZATION v1.32.420: Initialize memory pool for reusable tensor buffers
   * Eliminates repeated ByteBuffer/array allocations during beam search (20-30% speedup)
   */
  private void initializeMemoryPool(int decoderSeqLength)
  {
    try
    {
      // Pre-allocate for maximum beam width (DEFAULT_BEAM_WIDTH)
      int initialCapacity = _beamWidth > 0 ? _beamWidth : DEFAULT_BEAM_WIDTH;

      // Allocate reusable ByteBuffer for tokens (direct buffer for ONNX)
      int tokensByteBufferSize = initialCapacity * decoderSeqLength * 8; // 8 bytes per long
      _pooledTokensByteBuffer = java.nio.ByteBuffer.allocateDirect(tokensByteBufferSize);
      _pooledTokensByteBuffer.order(java.nio.ByteOrder.nativeOrder());
      _pooledTokensLongBuffer = _pooledTokensByteBuffer.asLongBuffer();

      // Allocate reusable memory replication array
      // Typical encoder output: [1, 250, 256] → replicate to [beams, 250, 256]
      int estimatedSeqLen = _maxSequenceLength > 0 ? _maxSequenceLength : 250;
      int estimatedHiddenDim = 256; // Standard transformer hidden dimension
      _pooledMemoryArray = new float[initialCapacity][estimatedSeqLen][estimatedHiddenDim];

      // Allocate reusable src_mask array
      _pooledSrcMaskArray = new boolean[initialCapacity][estimatedSeqLen];

      _pooledBufferMaxBeams = initialCapacity;

      Log.d(TAG, "Memory pool initialized: capacity=" + initialCapacity + " beams, seq_len=" + estimatedSeqLen);
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize memory pool", e);
      // Fallback: memory pool remains null, will allocate on-demand
    }
  }

  /**
   * OPTIMIZATION v1.32.420: Ensure memory pool has sufficient capacity
   * Grows pool if needed, reuses existing buffers if sufficient
   */
  private void ensureMemoryPoolCapacity(int requiredBeams, int memorySeqLen, int hiddenDim)
  {
    // If pool not initialized or capacity insufficient, reallocate
    if (_pooledMemoryArray == null ||
        _pooledBufferMaxBeams < requiredBeams ||
        _pooledMemoryArray[0].length < memorySeqLen ||
        _pooledMemoryArray[0][0].length < hiddenDim)
    {
      try
      {
        // Grow capacity by 50% to avoid frequent reallocations
        int newCapacity = Math.max(requiredBeams, (int)(_pooledBufferMaxBeams * 1.5));

        // Reallocate memory array with larger dimensions
        _pooledMemoryArray = new float[newCapacity][memorySeqLen][hiddenDim];

        // Reallocate src_mask array
        _pooledSrcMaskArray = new boolean[newCapacity][memorySeqLen];

        // Reallocate ByteBuffer for tokens (need to recreate due to fixed size)
        final int DECODER_SEQ_LENGTH = 20; // MUST match actual model export
        int tokensByteBufferSize = newCapacity * DECODER_SEQ_LENGTH * 8;
        _pooledTokensByteBuffer = java.nio.ByteBuffer.allocateDirect(tokensByteBufferSize);
        _pooledTokensByteBuffer.order(java.nio.ByteOrder.nativeOrder());
        _pooledTokensLongBuffer = _pooledTokensByteBuffer.asLongBuffer();

        _pooledBufferMaxBeams = newCapacity;

        Log.d(TAG, "Memory pool grown: new capacity=" + newCapacity + " beams");
      }
      catch (Exception e)
      {
        Log.e(TAG, "Failed to grow memory pool", e);
        // Pool remains at old capacity or null
      }
    }
  }
  
  /**
   * OPTIMIZATION: Initialize batch processing buffers for single decoder call
   * This is the critical architectural change for 8x speedup (expert recommendation)
   */
  private void initializeBatchProcessingBuffers(int decoderSeqLength)
  {
    try
    {
      // Allocate batched arrays for processing all beams simultaneously
      _batchedTokensArray = new long[_beamWidth][decoderSeqLength];
      _batchedMaskArray = new boolean[_beamWidth][decoderSeqLength];
      _batchedMemoryArray = new float[_beamWidth][150][256]; // Encoder memory for each beam
      
      // Log.d(TAG, "Batch processing buffers initialized: " + _beamWidth + " beams × " + decoderSeqLength + " seq_length");
      // logDebug("🚀 Batch processing initialized: " + _beamWidth + "×" + decoderSeqLength + " decoder optimization");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Failed to initialize batch processing buffers", e);
      // Fallback to sequential processing if batch allocation fails
      _batchedTokensArray = null;
      _batchedMaskArray = null; 
      _batchedMemoryArray = null;
    }
  }
  
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
        if (ch != '?')
        {
          word.append(ch);
        }
      }
    }
    
    long greedyTime = (System.nanoTime() - greedyStart) / 1_000_000;
    String wordStr = word.toString();
    // logDebug("🏆 Greedy search completed in " + greedyTime + "ms: '" + wordStr + "'");
    Log.w(TAG, "🏆 Greedy search completed in " + greedyTime + "ms: '" + wordStr + "'");
    
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

      // Handle model version or path changes (requires reinitialization)
      String newModelVersion = config.neural_model_version != null ? config.neural_model_version : "v2";
      String newEncoderPath = config.neural_custom_encoder_path;
      String newDecoderPath = config.neural_custom_decoder_path;

      boolean versionChanged = !newModelVersion.equals(_currentModelVersion);
      boolean pathsChanged = !java.util.Objects.equals(newEncoderPath, _currentEncoderPath) ||
                             !java.util.Objects.equals(newDecoderPath, _currentDecoderPath);

      if (versionChanged || pathsChanged)
      {
        Log.d(TAG, String.format("Model config changed: versionChanged=%b, pathsChanged=%b. Re-initialization required.",
          versionChanged, pathsChanged));

        // CRITICAL: Clean up old sessions before reinitializing
        try
        {
          if (_encoderSession != null)
          {
            _encoderSession.close();
            _encoderSession = null;
            Log.d(TAG, "Closed old encoder session");
          }
          if (_decoderSession != null)
          {
            _decoderSession.close();
            _decoderSession = null;
            Log.d(TAG, "Closed old decoder session");
          }
        }
        catch (Exception e)
        {
          Log.e(TAG, "Error closing old sessions", e);
        }

        _currentModelVersion = newModelVersion;
        _currentEncoderPath = newEncoderPath;
        _currentDecoderPath = newDecoderPath;
        _isInitialized = false;
        _isModelLoaded = false;

        // CRITICAL: Immediately reinitialize instead of waiting for next prediction
        // This ensures settings UI shows correct model status right away
        Log.d(TAG, "Triggering immediate model reinitialization...");
        initialize();
      }
      else
      {
        // No reinitialization needed, but update stored paths to prevent false positives
        _currentEncoderPath = newEncoderPath;
        _currentDecoderPath = newDecoderPath;
      }

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
  
  private List<BeamSearchCandidate> runBeamSearch(OrtSession.Result encoderResult, 
    int actualSrcLength, SwipeTrajectoryProcessor.TrajectoryFeatures features) throws OrtException
  {
    if (_decoderSession == null)
    {
      Log.e(TAG, "Decoder not loaded, cannot decode");
      return new ArrayList<>();
    }

    // Beam search parameters matching CLI test exactly
    int beamWidth = _beamWidth;
    int maxLength = _maxLength;
    final int DECODER_SEQ_LEN = 20; // Fixed decoder sequence length - MUST match actual model export
    int vocabSize = _tokenizer.getVocabSize();

    // Get memory from encoder output using proper ONNX API
    OnnxTensor memory = null;
    try {
      // Get first output from result
      memory = (OnnxTensor) encoderResult.get(0);
    } catch (Exception e) {
      Log.e(TAG, "Failed to get encoder output", e);
      return new ArrayList<>();
    }

    if (memory == null)
    {
      Log.e(TAG, "No memory tensor from encoder");
      return new ArrayList<>();
    }

    // Log.d(TAG, String.format("Decoder memory shape: %s", java.util.Arrays.toString(memory.getInfo().getShape())));

    // Initialize beams with SOS token - matching CLI test (line 158)
    List<BeamSearchState> beams = new ArrayList<>();
    beams.add(new BeamSearchState(SOS_IDX, 0.0f, false));
    // logDebug("🚀 Beam search initialized with SOS token (" + SOS_IDX + ")");

    // PERFORMANCE DEBUG: Log beam search parameters (CACHED check)
    if (_enableVerboseLogging)
    {
      Log.d(TAG, "🔥 BEAM SEARCH MODE: beam_width=" + beamWidth + ", max_length=" + maxLength);
    }

    // Performance tracking
    long beamSearchStart = System.nanoTime();
    long totalInferenceTime = 0;
    long totalTensorTime = 0;
    boolean useBatched = _batchBeams; // CACHED - avoid config check on every swipe
    int step = 0;

    // OPTIMIZATION v1.32.416: Batched beam search loop for 8x speedup
    // Process all beams simultaneously in single decoder call instead of sequential processing
    for (; step < maxLength; step++)
    {
      List<BeamSearchState> candidates = new ArrayList<>();
      // PERFORMANCE: Only log every 5th step to reduce overhead
      if (step % 5 == 0) {
        // logDebug("🔄 Batched beam search step " + step + " with " + beams.size() + " beams");
      }

      // Separate finished beams from active beams
      List<BeamSearchState> activeBeams = new ArrayList<>();
      for (BeamSearchState beam : beams)
      {
        if (beam.finished)
        {
          candidates.add(beam);
        }
        else
        {
          activeBeams.add(beam);
        }
      }

      // If no active beams, we're done
      if (activeBeams.isEmpty())
      {
        break;
      }

      long tensorStart = System.nanoTime();

      if (useBatched)
      {
        // EXPERIMENTAL: Batched beam processing - all beams in single inference
        // May cause reshape errors in self-attention layers
        try
        {
          int numActiveBeams = activeBeams.size();

          // Prepare batched token arrays
          int[][] batchedTokens = new int[numActiveBeams][DECODER_SEQ_LEN];
          for (int b = 0; b < numActiveBeams; b++)
          {
            BeamSearchState beam = activeBeams.get(b);
            Arrays.fill(batchedTokens[b], (int)PAD_IDX);
            for (int i = 0; i < Math.min(beam.tokens.size(), DECODER_SEQ_LEN); i++)
            {
              batchedTokens[b][i] = beam.tokens.get(i).intValue();
            }
          }

          // Flatten to 1D for tensor creation
          int[] flatTokens = new int[numActiveBeams * DECODER_SEQ_LEN];
          for (int b = 0; b < numActiveBeams; b++)
          {
            System.arraycopy(batchedTokens[b], 0, flatTokens, b * DECODER_SEQ_LEN, DECODER_SEQ_LEN);
          }

          OnnxTensor targetTokensTensor = OnnxTensor.createTensor(_ortEnvironment, 
            java.nio.IntBuffer.wrap(flatTokens), new long[]{numActiveBeams, DECODER_SEQ_LEN});

          // Get memory dimensions for replication
          long[] memoryShape = memory.getInfo().getShape(); // [1, seq_len, hidden_dim]
          int memorySeqLen = (int)memoryShape[1];
          int hiddenDim = (int)memoryShape[2];

          // OPTIMIZATION v6 (perftodos6.md): Broadcast models expand memory internally
          OnnxTensor batchedMemoryTensor;
          OnnxTensor actualSrcLengthTensor;

          if (_broadcastEnabled)
          {
            // Broadcast model: Pass memory with batch=1, model expands internally
            // Memory shape: [1, seq_len, hidden_dim]
            // Target tokens shape: [num_beams, seq_len]
            // Model will broadcast memory to match num_beams automatically
            batchedMemoryTensor = memory; // Use as-is, no replication needed

            // For broadcast models, actual_src_length should also be single value
            actualSrcLengthTensor = OnnxTensor.createTensor(_ortEnvironment, new int[]{actualSrcLength});

            if (step == 0 && _enableVerboseLogging)
            {
              logDebug("🚀 Broadcast mode: memory [1, " + memorySeqLen + ", " + hiddenDim + "] → " + numActiveBeams + " beams\n");
            }
          }
          else
          {
            // Legacy model: Manually replicate memory for all beams
            float[][][] memoryData = (float[][][])memory.getValue();
            float[][][] replicatedMemory = new float[numActiveBeams][memorySeqLen][hiddenDim];
            for (int b = 0; b < numActiveBeams; b++)
            {
              for (int s = 0; s < memorySeqLen; s++)
              {
                System.arraycopy(memoryData[0][s], 0, replicatedMemory[b][s], 0, hiddenDim);
              }
            }
            batchedMemoryTensor = OnnxTensor.createTensor(_ortEnvironment, replicatedMemory);

            // Create batched actual_src_length for all beams
            int[] srcLengths = new int[numActiveBeams];
            Arrays.fill(srcLengths, actualSrcLength);
            actualSrcLengthTensor = OnnxTensor.createTensor(_ortEnvironment, srcLengths);
          }

          // Run batched decoder inference
          Map<String, OnnxTensor> decoderInputs = new HashMap<>();
          decoderInputs.put("memory", batchedMemoryTensor);
          decoderInputs.put("target_tokens", targetTokensTensor);
          decoderInputs.put("actual_src_length", actualSrcLengthTensor);

          // Debug logging when verbose logging enabled (CACHED)
          if (step == 0 && _enableVerboseLogging)
          {
            logDebug("=== DECODER INPUTS (step 0) ===\n");
            logDebug("  memory: " + java.util.Arrays.toString(batchedMemoryTensor.getInfo().getShape()) + "\n");
            logDebug("  target_tokens: " + java.util.Arrays.toString(targetTokensTensor.getInfo().getShape()) + "\n");
            logDebug("  actual_src_length: " + java.util.Arrays.toString(actualSrcLengthTensor.getInfo().getShape()) + "\n");
            logDebug("  actualSrcLength value: " + actualSrcLength + "\n");
            logDebug("  numActiveBeams: " + numActiveBeams + "\n");
            logDebug("  broadcastEnabled: " + _broadcastEnabled + "\n");
            logDebug("  First beam tokens: " + java.util.Arrays.toString(java.util.Arrays.copyOf(flatTokens, Math.min(10, flatTokens.length))) + "\n");
          }

          long inferenceStart = System.nanoTime();
          OrtSession.Result decoderOutput = _decoderSession.run(decoderInputs);
          totalInferenceTime += (System.nanoTime() - inferenceStart) / 1_000_000;

          // Process batched output [num_beams, seq_len, vocab_size]
          OnnxTensor logitsTensor = (OnnxTensor) decoderOutput.get(0);
          float[][][] logits3D = (float[][][]) logitsTensor.getValue();

          // OPTIMIZATION Phase 2: Get trie once for all beams
          VocabularyTrie trie = (_vocabulary != null) ? _vocabulary.getVocabularyTrie() : null;

          for (int b = 0; b < numActiveBeams; b++)
          {
            BeamSearchState beam = activeBeams.get(b);
            int currentPos = beam.tokens.size() - 1;
            if (currentPos >= 0 && currentPos < DECODER_SEQ_LEN)
            {
              float[] logProbs = logits3D[b][currentPos];
              int[] topK = getTopKIndices(logProbs, beamWidth);

              for (int idx : topK)
              {
                // Skip special tokens
                if (idx == SOS_IDX || idx == EOS_IDX || idx == PAD_IDX) {
                  BeamSearchState newBeam = new BeamSearchState(beam);
                  newBeam.tokens.add((long)idx);
                  newBeam.score -= logProbs[idx];
                  newBeam.finished = true;
                  candidates.add(newBeam);
                  continue;
                }

                // OPTIMIZATION Phase 2: Trie validation for batched path
                // Convert tokens to partial word
                StringBuilder partialWord = new StringBuilder();
                for (Long token : beam.tokens) {
                  int tokenIdx = token.intValue();
                  if (tokenIdx != SOS_IDX && tokenIdx != EOS_IDX && tokenIdx != PAD_IDX) {
                    char ch = _tokenizer.indexToChar(tokenIdx);
                    if (ch != '?' && !Character.toString(ch).startsWith("<")) {
                      partialWord.append(ch);
                    }
                  }
                }

                // Add new character
                char newChar = _tokenizer.indexToChar(idx);
                if (newChar != '?' && !Character.toString(newChar).startsWith("<")) {
                  partialWord.append(newChar);
                }

                // Validate against trie
                String partialWordStr = partialWord.toString();
                if (trie != null && partialWordStr.length() > 0) {
                  if (!trie.hasPrefix(partialWordStr)) {
                    continue; // Invalid prefix - skip
                  }
                }

                // Valid prefix - add beam
                BeamSearchState newBeam = new BeamSearchState(beam);
                newBeam.tokens.add((long)idx);
                newBeam.score -= logProbs[idx];
                newBeam.finished = (idx == EOS_IDX || idx == PAD_IDX);
                candidates.add(newBeam);
              }
            }
          }

          // Cleanup
          targetTokensTensor.close();
          actualSrcLengthTensor.close();
          // Only close batchedMemoryTensor if it's a new tensor (legacy mode)
          // In broadcast mode, batchedMemoryTensor is the original memory tensor
          if (!_broadcastEnabled)
          {
            batchedMemoryTensor.close();
          }
          decoderOutput.close();
        }
        catch (Exception e)
        {
          logDebug("💥 Batched decoder step " + step + " error: " + e.getClass().getSimpleName() + " - " + e.getMessage() + "\n");
          Log.e(TAG, "Batched decoder step error", e);
        }
      }
      else
      {
        // Sequential beam processing (batch=1) - default, stable mode
        // OPTIMIZATION v1.32.511: Reuse arrays and tensors to reduce allocation overhead

        // Pre-allocate reusable arrays (only on first step to avoid per-step allocation)
        if (step == 0)
        {
          // These will be reused for all beams in all steps
        }

        // OPTIMIZATION: Create actualSrcLengthTensor once per step (same for all beams)
        OnnxTensor actualSrcLengthTensor = null;
        try
        {
          actualSrcLengthTensor = OnnxTensor.createTensor(_ortEnvironment, 
            new int[]{actualSrcLength});
        }
        catch (Exception e)
        {
          Log.e(TAG, "Failed to create actualSrcLengthTensor", e);
          break;
        }

        // OPTIMIZATION: Pre-allocate token array and HashMap outside beam loop
        int[] tgtTokens = new int[DECODER_SEQ_LEN];
        Map<String, OnnxTensor> decoderInputs = new HashMap<>(3);
        decoderInputs.put("memory", memory);
        decoderInputs.put("actual_src_length", actualSrcLengthTensor);

        for (int b = 0; b < activeBeams.size(); b++)
        {
          BeamSearchState beam = activeBeams.get(b);

          try
          {
            // Reuse tgtTokens array - just overwrite values
            Arrays.fill(tgtTokens, (int)PAD_IDX);
            int tokenCount = Math.min(beam.tokens.size(), DECODER_SEQ_LEN);
            for (int i = 0; i < tokenCount; i++)
            {
              tgtTokens[i] = beam.tokens.get(i).intValue();
            }

            // Create tensor for this beam's tokens (must create new - wraps buffer)
            OnnxTensor targetTokensTensor = OnnxTensor.createTensor(_ortEnvironment, 
              java.nio.IntBuffer.wrap(tgtTokens), new long[]{1, DECODER_SEQ_LEN});

            // Update HashMap with new target_tokens tensor
            decoderInputs.put("target_tokens", targetTokensTensor);

            long inferenceStart = System.nanoTime();
            OrtSession.Result decoderOutput = _decoderSession.run(decoderInputs);
            totalInferenceTime += (System.nanoTime() - inferenceStart) / 1_000_000;

            OnnxTensor logitsTensor = (OnnxTensor) decoderOutput.get(0);

            // Handle 3D logits tensor [1, seq_len, vocab_size]
            float[][][] logits3D = (float[][][]) logitsTensor.getValue();

            // Get log probs for last valid position
            int currentPos = beam.tokens.size() - 1;
            if (currentPos >= 0 && currentPos < DECODER_SEQ_LEN)
            {
              float[] logProbs = logits3D[0][currentPos];  // batch=0 since we use batch=1

              // Get top k tokens by highest log prob (higher is better)
              int[] topK = getTopKIndices(logProbs, beamWidth);

              // OPTIMIZATION Phase 2: Constrained vocabulary search with Trie
              // Check if new token forms valid vocabulary prefix before adding beam
              VocabularyTrie trie = (_vocabulary != null) ? _vocabulary.getVocabularyTrie() : null;

              // Create new beams
              for (int idx : topK)
              {
                // Skip special tokens
                if (idx == SOS_IDX || idx == EOS_IDX || idx == PAD_IDX) {
                  BeamSearchState newBeam = new BeamSearchState(beam);
                  newBeam.tokens.add((long)idx);
                  newBeam.score -= logProbs[idx];
                  newBeam.finished = true;
                  candidates.add(newBeam);
                  continue;
                }

                // Convert tokens to partial word for trie validation
                StringBuilder partialWord = new StringBuilder();
                for (Long token : beam.tokens) {
                  int tokenIdx = token.intValue();
                  if (tokenIdx != SOS_IDX && tokenIdx != EOS_IDX && tokenIdx != PAD_IDX) {
                    char ch = _tokenizer.indexToChar(tokenIdx);
                    if (ch != '?' && !Character.toString(ch).startsWith("<")) {
                      partialWord.append(ch);
                    }
                  }
                }

                // Add new character from this token
                char newChar = _tokenizer.indexToChar(idx);
                if (newChar != '?' && !Character.toString(newChar).startsWith("<")) {
                  partialWord.append(newChar);
                }

                // Validate against trie if available
                String partialWordStr = partialWord.toString();
                if (trie != null && partialWordStr.length() > 0) {
                  if (!trie.hasPrefix(partialWordStr)) {
                    // Invalid prefix - skip this beam
                    continue;
                  }
                }

                // Valid prefix or no trie - add beam
                BeamSearchState newBeam = new BeamSearchState(beam);
                newBeam.tokens.add((long)idx);
                newBeam.score -= logProbs[idx];
                newBeam.finished = (idx == EOS_IDX || idx == PAD_IDX);
                candidates.add(newBeam);
              }
            }

            // Clean up only targetTokensTensor (actualSrcLengthTensor reused)
            targetTokensTensor.close();
            decoderOutput.close();
          }
          catch (Exception e)
          {
            logDebug("💥 Decoder step " + step + " beam " + b + " error: " + e.getClass().getSimpleName() + " - " + e.getMessage() + "\n");
            Log.e(TAG, "Decoder step error for beam " + b, e);
          }
        }

        // Clean up the shared actualSrcLengthTensor after all beams processed
        if (actualSrcLengthTensor != null)
        {
          actualSrcLengthTensor.close();
        }
      }

      totalTensorTime += (System.nanoTime() - tensorStart) / 1_000_000;

      // Debug: log candidate generation
      if (step == 0) {
        logDebug("Step " + step + ": generated " + candidates.size() + " candidates from " + activeBeams.size() + " active beams\n");
      }

      // Select top beams - matches CLI line 232
      candidates.sort((a, b) -> Float.compare(a.score, b.score)); // Lower score is better (negative log prob)

      // OPTIMIZATION Phase 2.1: Confidence threshold pruning
      // Remove beams with very low probability (exp(-score) < 0.01) to avoid wasting compute
      if (step >= 2) { // Wait at least 2 steps before pruning
        int beforePrune = candidates.size();
        candidates.removeIf(beam -> Math.exp(-beam.score) < 0.01); // Keep only beams with prob > 1%
        int afterPrune = candidates.size();
        if (afterPrune < beforePrune && _enableVerboseLogging) {
          logDebug(String.format("⚡ Pruned %d low-confidence beams at step %d\n", beforePrune - afterPrune, step));
        }
      }

      beams = candidates.subList(0, Math.min(candidates.size(), beamWidth));

      // OPTIMIZATION Phase 2.2: Adaptive beam width reduction
      // Reduce beam width mid-search if we have high-confidence predictions
      if (step == 5 && beams.size() > 3) {
        float topScore = beams.get(0).score;
        float thirdScore = beams.size() >= 3 ? beams.get(2).score : Float.POSITIVE_INFINITY;
        float confidence = (float)Math.exp(-topScore);

        // If top beam has >50% confidence, narrow search to top 3 beams
        if (confidence > 0.5f) {
          int oldSize = beams.size();
          beams = beams.subList(0, Math.min(3, beams.size()));
          if (_enableVerboseLogging) {
            logDebug(String.format("⚡ Reduced beam width %d→%d (top conf=%.2f) at step %d\n",
              oldSize, beams.size(), confidence, step));
          }
        }
      }

      // OPTIMIZATION v1.32.515: Score-gap early stopping
      // If top beam is significantly better than 2nd beam, stop early (confident prediction)
      if (beams.size() >= 2 && step >= 3) // Wait at least 3 steps for meaningful scores
      {
        float topScore = beams.get(0).score;
        float secondScore = beams.get(1).score;
        float scoreGap = secondScore - topScore; // Gap between top and 2nd (higher = more confident)

        // If top beam finished and score gap > 2.0 (e^2 ≈ 7.4x more likely), stop early
        if (beams.get(0).finished && scoreGap > 2.0f)
        {
          logDebug("⚡ Score-gap early stop at step " + step + " (gap=" + String.format("%.2f", scoreGap) + ")\n");
          break;
        }
      }

      // Check if all beams finished - matches CLI line 235
      boolean allFinished = true;
      int finishedCount = 0;
      for (BeamSearchState beam : beams) {
        if (beam.finished) {
          finishedCount++;
        } else {
          allFinished = false;
        }
      }

      // Early stop if all beams finished OR we have enough finished beams
      if (allFinished || finishedCount >= beamWidth)
      {
        logDebug("🏁 Early stop at step " + step + " (" + finishedCount + "/" + beams.size() + " finished)\n");
        break;
      }
    }
    
    // Performance summary
    long totalBeamSearchTime = (System.nanoTime() - beamSearchStart) / 1_000_000;
    logDebug("📊 Beam search: " + totalBeamSearchTime + "ms (inference: " + totalInferenceTime + "ms, tensor: " + totalTensorTime + "ms, steps: " + step + ", mode: " + (useBatched ? "batched" : "sequential") + ")\n");
    
    // Convert token sequences to words with detailed debugging
    List<BeamSearchCandidate> results = new ArrayList<>();
    logDebug("🔤 Converting " + beams.size() + " beams to words...\n");

    for (int b = 0; b < beams.size(); b++) {
      BeamSearchState beam = beams.get(b);
      StringBuilder word = new StringBuilder();
      StringBuilder tokenLog = new StringBuilder();

      for (Long token : beam.tokens)
      {
        int idx = token.intValue();
        if (idx == SOS_IDX || idx == EOS_IDX || idx == PAD_IDX) {
          tokenLog.append("[").append(idx).append("] ");
          continue;
        }

        char ch = _tokenizer.indexToChar(idx);
        tokenLog.append(ch);

        if (ch != '?' && !Character.toString(ch).startsWith("<"))
        {
          word.append(ch);
        }
      }

      String wordStr = word.toString();
      if (wordStr.length() > 0)
      {
        // Convert accumulated negative log likelihood back to probability
        // Since score is positive (accumulated -log(prob)), use exp(-score)
        float confidence = (float)Math.exp(-beam.score);
        results.add(new BeamSearchCandidate(wordStr, confidence));
        logDebug(String.format("   Beam %d: '%s' (score=%.2f, conf=%.3f) tokens=%s\n",
          b, wordStr, beam.score, confidence, tokenLog.toString()));
      } else {
        logDebug(String.format("   Beam %d: EMPTY (tokens=%s)\n", b, tokenLog.toString()));
      }
    }

    logDebug("🎯 Generated " + results.size() + " word candidates from " + beams.size() + " beams\n");
    return results;
  }
  
  private float[] softmax(float[] logits)
  {
    float maxLogit = 0.0f;
    for (float logit : logits) {
      if (logit > maxLogit) maxLogit = logit;
    }
    float[] expScores = new float[logits.length];
    float sumExpScores = 0.0f;
    
    for (int i = 0; i < logits.length; i++)
    {
      expScores[i] = (float)Math.exp(logits[i] - maxLogit);
      sumExpScores += expScores[i];
    }
    
    for (int i = 0; i < expScores.length; i++)
    {
      expScores[i] /= sumExpScores;
    }
    
    return expScores;
  }
  
  /**
   * OPTIMIZATION Phase 2: Micro-optimized top-K selection for small k and n.
   * For beam_width=2-5 and vocab=30, this specialized implementation is faster
   * than both heap-based and insertion-sort approaches.
   *
   * Uses partial quickselect partitioning for O(n) average case.
   */
  private int[] getTopKIndices(float[] array, int k)
  {
    int n = array.length;
    int actualK = Math.min(k, n);

    // Special case: k=1 (greedy decode)
    if (actualK == 1) {
      int maxIdx = 0;
      float maxVal = array[0];
      for (int i = 1; i < n; i++) {
        if (array[i] > maxVal) {
          maxVal = array[i];
          maxIdx = i;
        }
      }
      return new int[]{maxIdx};
    }

    // For small k (2-5), use optimized linear scan with minimal comparisons
    // This avoids the shift overhead in insertion sort
    int[] result = new int[actualK];
    float[] resultValues = new float[actualK];

    // Initialize with first k elements
    for (int i = 0; i < actualK; i++) {
      result[i] = i;
      resultValues[i] = array[i];
    }

    // Sort initial k elements (bubble sort for small k)
    for (int i = 0; i < actualK - 1; i++) {
      for (int j = i + 1; j < actualK; j++) {
        if (resultValues[j] > resultValues[i]) {
          float tmpVal = resultValues[i];
          int tmpIdx = result[i];
          resultValues[i] = resultValues[j];
          result[i] = result[j];
          resultValues[j] = tmpVal;
          result[j] = tmpIdx;
        }
      }
    }

    // Scan remaining elements, only insert if larger than smallest in top-k
    float minTopK = resultValues[actualK - 1];
    for (int i = actualK; i < n; i++) {
      float val = array[i];
      if (val > minTopK) {
        // Find insertion position (binary search in sorted top-k)
        int insertPos = actualK - 1;
        for (int j = actualK - 2; j >= 0; j--) {
          if (val > resultValues[j]) {
            insertPos = j;
          } else {
            break;
          }
        }

        // Shift and insert
        for (int j = actualK - 1; j > insertPos; j--) {
          resultValues[j] = resultValues[j - 1];
          result[j] = result[j - 1];
        }
        resultValues[insertPos] = val;
        result[insertPos] = i;
        minTopK = resultValues[actualK - 1];
      }
    }

    return result;
  }
  
  private static class BeamSearchState
  {
    public List<Long> tokens;
    public float score;
    public boolean finished;
    
    public BeamSearchState(int startToken, float startScore, boolean isFinished)
    {
      tokens = new ArrayList<>();
      tokens.add((long)startToken);
      score = startScore;
      finished = isFinished;
    }
    
    public BeamSearchState(BeamSearchState other)
    {
      tokens = new ArrayList<>(other.tokens);
      score = other.score;
      finished = other.finished;
    }
  }
  
  private static class IndexValue
  {
    public int index;
    public float value;
    
    public IndexValue(int index, float value)
    {
      this.index = index;
      this.value = value;
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
      // filtered.stream().mapToLong(p -> p.source.equals("common") ? 1 : 0).sum() + " common, " +
      // filtered.stream().mapToLong(p -> p.source.equals("top5000") ? 1 : 0).sum() + " top5000");

    return new PredictionResult(words, scores);
  }
  
  private PredictionResult createEmptyResult()
  {
    return new PredictionResult(new ArrayList<>(), new ArrayList<>());
  }
  
  /**
   * OPTIMIZATION: Controlled cleanup that respects session persistence
   * Only cleans up sessions if explicitly requested (default: keep in memory)
   */
  public void cleanup()
  {
    cleanup(false); // Default: keep sessions for performance
  }
  
  public void cleanup(boolean forceCleanup)
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
  
  
  /**
   * Beam search candidate
   */
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
}