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
import java.util.List;
import java.util.Map;

/**
 * ONNX-based neural swipe predictor using transformer encoder-decoder architecture
 * Replaces legacy DTW/Bayesian prediction with state-of-the-art neural networks
 */
public class OnnxSwipePredictor
{
  private static final String TAG = "OnnxSwipePredictor";
  
  // Model configuration matching web demo exactly
  private static final int MAX_SEQUENCE_LENGTH = 150; // Must match web demo training
  private static final int TRAJECTORY_FEATURES = 6; // x, y, vx, vy, ax, ay
  private static final float NORMALIZED_WIDTH = 1.0f;
  private static final float NORMALIZED_HEIGHT = 1.0f;
  
  // Beam search parameters
  private static final int DEFAULT_BEAM_WIDTH = 8;
  private static final int DEFAULT_MAX_LENGTH = 35;
  private static final float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f;
  
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
  private NeuralVocabulary _vocabulary;
  
  // Model state
  private boolean _isModelLoaded = false;
  private boolean _isInitialized = false;
  
  // Configuration parameters
  private int _beamWidth = DEFAULT_BEAM_WIDTH;
  private int _maxLength = DEFAULT_MAX_LENGTH;
  private float _confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD;
  
  // Debug logging
  private NeuralSwipeTypingEngine.DebugLogger _debugLogger;
  
  public OnnxSwipePredictor(Context context)
  {
    _context = context;
    _ortEnvironment = OrtEnvironment.getEnvironment();
    _trajectoryProcessor = new SwipeTrajectoryProcessor();
    _tokenizer = new SwipeTokenizer();
    _vocabulary = new NeuralVocabulary();
    
    Log.d(TAG, "OnnxSwipePredictor initialized");
  }
  
  /**
   * Initialize the predictor with models from assets
   */
  public boolean initialize()
  {
    if (_isInitialized)
    {
      return _isModelLoaded;
    }
    
    try
    {
      Log.d(TAG, "Loading ONNX models...");
      logDebug("🔄 Loading ONNX transformer models...");
      
      // Load encoder model (using correct name from web demo)
      byte[] encoderModelData = loadModelFromAssets("models/swipe_model_character_quant.onnx");
      if (encoderModelData != null)
      {
        logDebug("📥 Encoder model data loaded: " + encoderModelData.length + " bytes");
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        _encoderSession = _ortEnvironment.createSession(encoderModelData, sessionOptions);
        logDebug("✅ Encoder session created successfully");
        logDebug("   Inputs: " + _encoderSession.getInputNames());
        logDebug("   Outputs: " + _encoderSession.getOutputNames());
        Log.d(TAG, "Encoder model loaded successfully");
      }
      else
      {
        logDebug("❌ Failed to load encoder model data");
        Log.e(TAG, "Failed to load encoder model data");
      }
      
      // Load decoder model (using correct name from web demo)
      byte[] decoderModelData = loadModelFromAssets("models/swipe_decoder_character_quant.onnx");
      if (decoderModelData != null)
      {
        logDebug("📥 Decoder model data loaded: " + decoderModelData.length + " bytes");
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        _decoderSession = _ortEnvironment.createSession(decoderModelData, sessionOptions);
        logDebug("✅ Decoder session created successfully");
        logDebug("   Inputs: " + _decoderSession.getInputNames());
        logDebug("   Outputs: " + _decoderSession.getOutputNames());
        Log.d(TAG, "Decoder model loaded successfully");
      }
      else
      {
        logDebug("❌ Failed to load decoder model data");
        Log.e(TAG, "Failed to load decoder model data");
      }
      
      // Load tokenizer configuration
      boolean tokenizerLoaded = _tokenizer.loadFromAssets(_context);
      logDebug("📝 Tokenizer loaded: " + tokenizerLoaded + " (vocab size: " + _tokenizer.getVocabSize() + ")");
      
      // Load vocabulary with multi-level caching
      boolean vocabularyLoaded = _vocabulary.loadVocabulary();
      logDebug("📚 Vocabulary loaded: " + vocabularyLoaded + " (words: " + _vocabulary.getVocabularySize() + ")");
      
      _isModelLoaded = (_encoderSession != null && _decoderSession != null);
      _isInitialized = true;
      
      if (_isModelLoaded)
      {
        logDebug("🧠 ONNX neural prediction system ready!");
        Log.d(TAG, "ONNX neural prediction system ready");
      }
      else
      {
        logDebug("⚠️ ONNX models failed to load - missing encoder or decoder session");
        Log.w(TAG, "Failed to load ONNX models");
      }
      
      return _isModelLoaded;
    }
    catch (Exception e)
    {
      logDebug("💥 ONNX initialization exception: " + e.getClass().getSimpleName() + " - " + e.getMessage());
      Log.e(TAG, "Failed to initialize ONNX models", e);
      _isInitialized = true;
      _isModelLoaded = false;
      return false;
    }
  }
  
  /**
   * Predict words from swipe input using neural transformer
   */
  public PredictionResult predict(SwipeInput input)
  {
    if (!_isModelLoaded)
    {
      throw new RuntimeException("ONNX models failed to load: " + 
        (_encoderSession == null ? "encoder missing " : "") +
        (_decoderSession == null ? "decoder missing" : ""));
    }
    
    try
    {
      Log.d(TAG, "Neural prediction for swipe with " + input.coordinates.size() + " points");
      logDebug("🚀 Starting neural prediction for " + input.coordinates.size() + " points");
      
      // Extract trajectory features
      SwipeTrajectoryProcessor.TrajectoryFeatures features = 
        _trajectoryProcessor.extractFeatures(input, MAX_SEQUENCE_LENGTH);
      
      // Run encoder inference with proper ONNX API
      OnnxTensor trajectoryTensor = null;
      OnnxTensor nearestKeysTensor = null; 
      OnnxTensor srcMaskTensor = null;
      OrtSession.Result encoderResult = null;
      
      try {
        trajectoryTensor = createTrajectoryTensor(features);
        nearestKeysTensor = createNearestKeysTensor(features);
        srcMaskTensor = createSourceMaskTensor(features);
        
        // Log tensor shapes for debugging
        logDebug("🔧 Encoder input tensor shapes:");
        logDebug("   trajectory_features: " + java.util.Arrays.toString(trajectoryTensor.getInfo().getShape()));
        logDebug("   nearest_keys: " + java.util.Arrays.toString(nearestKeysTensor.getInfo().getShape()));
        logDebug("   src_mask: " + java.util.Arrays.toString(srcMaskTensor.getInfo().getShape()) + " (BOOL)");
        
        Map<String, OnnxTensor> encoderInputs = new HashMap<>();
        encoderInputs.put("trajectory_features", trajectoryTensor);
        encoderInputs.put("nearest_keys", nearestKeysTensor);
        encoderInputs.put("src_mask", srcMaskTensor);
        
        // Run encoder inference with try-with-resources
        try (OrtSession.Result encoderResults = _encoderSession.run(encoderInputs)) {
          // Run beam search decoding, passing the original source mask
          List<BeamSearchCandidate> candidates = runBeamSearch(encoderResults, srcMaskTensor, features);
          
          // Convert to prediction result
          return createPredictionResult(candidates);
        }
        
      } finally {
        // Proper memory cleanup
        if (trajectoryTensor != null) trajectoryTensor.close();
        if (nearestKeysTensor != null) nearestKeysTensor.close();
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
   * Set configuration parameters
   */
  public void setConfig(Config config)
  {
    _config = config;
    
    // Update neural parameters from config
    if (config != null)
    {
      _beamWidth = config.neural_beam_width != 0 ? config.neural_beam_width : DEFAULT_BEAM_WIDTH;
      _maxLength = config.neural_max_length != 0 ? config.neural_max_length : DEFAULT_MAX_LENGTH;
      _confidenceThreshold = config.neural_confidence_threshold != 0 ? 
        config.neural_confidence_threshold : DEFAULT_CONFIDENCE_THRESHOLD;
    }
    
    Log.d(TAG, String.format("Neural config: beam_width=%d, max_length=%d, threshold=%.3f",
      _beamWidth, _maxLength, _confidenceThreshold));
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
   * Clean up resources
   */
  public void cleanup()
  {
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
      
      Log.d(TAG, "ONNX resources cleaned up");
    }
    catch (Exception e)
    {
      Log.e(TAG, "Error during cleanup", e);
    }
  }
  
  private byte[] loadModelFromAssets(String modelPath)
  {
    try
    {
      Log.d(TAG, "Loading ONNX model: " + modelPath);
      InputStream inputStream = _context.getAssets().open(modelPath);
      int available = inputStream.available();
      Log.d(TAG, "Model file size: " + available + " bytes");
      
      byte[] modelData = new byte[available];
      int totalRead = 0;
      while (totalRead < available) {
        int read = inputStream.read(modelData, totalRead, available - totalRead);
        if (read == -1) break;
        totalRead += read;
      }
      inputStream.close();
      
      Log.d(TAG, "Successfully loaded " + totalRead + " bytes from " + modelPath);
      return modelData;
    }
    catch (IOException e)
    {
      Log.e(TAG, "Failed to load model from assets: " + modelPath, e);
      return null;
    }
  }
  
  private OnnxTensor createTrajectoryTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features) 
    throws OrtException
  {
    // Create direct buffer as recommended by ONNX docs
    java.nio.ByteBuffer byteBuffer = java.nio.ByteBuffer.allocateDirect(MAX_SEQUENCE_LENGTH * TRAJECTORY_FEATURES * 4); // 4 bytes per float
    byteBuffer.order(java.nio.ByteOrder.nativeOrder());
    java.nio.FloatBuffer buffer = byteBuffer.asFloatBuffer();
    
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
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
    long[] shape = {1, MAX_SEQUENCE_LENGTH, TRAJECTORY_FEATURES};
    return OnnxTensor.createTensor(_ortEnvironment, buffer, shape);
  }
  
  private OnnxTensor createNearestKeysTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    // Create direct buffer as recommended by ONNX docs
    java.nio.ByteBuffer byteBuffer = java.nio.ByteBuffer.allocateDirect(MAX_SEQUENCE_LENGTH * 8); // 8 bytes per long
    byteBuffer.order(java.nio.ByteOrder.nativeOrder());
    java.nio.LongBuffer buffer = byteBuffer.asLongBuffer();
    
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
    {
      if (i < features.nearestKeys.size())
      {
        char key = features.nearestKeys.get(i);
        buffer.put(_tokenizer.charToIndex(key));
      }
      else
      {
        buffer.put(PAD_IDX); // Padding
      }
    }
    
    buffer.rewind();
    long[] shape = {1, MAX_SEQUENCE_LENGTH};
    return OnnxTensor.createTensor(_ortEnvironment, buffer, shape);
  }
  
  private OnnxTensor createSourceMaskTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    // Create 2D boolean array for proper tensor shape [1, MAX_SEQUENCE_LENGTH]
    boolean[][] maskData = new boolean[1][MAX_SEQUENCE_LENGTH];
    
    // Mask padded positions (true = masked/padded, false = valid)
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
    {
      maskData[0][i] = (i >= features.actualLength);
    }
    
    // Use 2D boolean array - ONNX API will infer shape as [1, 100]
    return OnnxTensor.createTensor(_ortEnvironment, maskData);
  }
  
  private List<BeamSearchCandidate> runBeamSearch(OrtSession.Result encoderResult, 
    OnnxTensor srcMaskTensor, SwipeTrajectoryProcessor.TrajectoryFeatures features) throws OrtException
  {
    if (_decoderSession == null)
    {
      Log.e(TAG, "Decoder not loaded, cannot decode");
      return new ArrayList<>();
    }
    
    // Beam search parameters matching web demo
    int beamWidth = _beamWidth;
    int maxLength = _maxLength;
    int decoderSeqLength = 20; // Fixed decoder sequence length
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
    
    Log.d(TAG, String.format("Decoder memory shape: %s", java.util.Arrays.toString(memory.getInfo().getShape())));
    
    // Initialize beams with SOS token
    List<BeamSearchState> beams = new ArrayList<>();
    beams.add(new BeamSearchState(SOS_IDX, 0.0f, false));
    
    // Beam search loop
    for (int step = 0; step < maxLength; step++)
    {
      List<BeamSearchState> candidates = new ArrayList<>();
      
      for (BeamSearchState beam : beams)
      {
        if (beam.finished)
        {
          candidates.add(beam);
          continue;
        }
        
        try
        {
          // Prepare decoder input - pad tokens to fixed length
          long[] paddedTokens = new long[decoderSeqLength];
          for (int i = 0; i < Math.min(beam.tokens.size(), decoderSeqLength); i++)
          {
            paddedTokens[i] = beam.tokens.get(i);
          }
          // Pad rest with PAD_IDX
          for (int i = beam.tokens.size(); i < decoderSeqLength; i++)
          {
            paddedTokens[i] = PAD_IDX;
          }
          
          // Create target mask 2D boolean array - true for PADDED positions, false for real tokens
          boolean[][] targetMaskData = new boolean[1][decoderSeqLength];
          for (int i = 0; i < decoderSeqLength; i++)
          {
            targetMaskData[0][i] = (i >= beam.tokens.size());
          }
          
          // Create tensors with ONNX Runtime 1.20.0
          OnnxTensor targetTokensTensor = OnnxTensor.createTensor(_ortEnvironment, 
            java.nio.LongBuffer.wrap(paddedTokens), new long[]{1, decoderSeqLength});
          OnnxTensor targetMaskTensor = OnnxTensor.createTensor(_ortEnvironment, targetMaskData);
          // Reuse the original srcMaskTensor from encoder (don't create new one)
          
          // Log decoder tensor shapes
          logDebug("🔧 Decoder input tensor shapes:");
          logDebug("   memory: " + java.util.Arrays.toString(memory.getInfo().getShape()));
          logDebug("   target_tokens: " + java.util.Arrays.toString(targetTokensTensor.getInfo().getShape()));
          logDebug("   target_mask: " + java.util.Arrays.toString(targetMaskTensor.getInfo().getShape()) + " (BOOL)");
          logDebug("   src_mask: " + java.util.Arrays.toString(srcMaskTensor.getInfo().getShape()) + " (BOOL - reused)");
          
          // Run decoder
          Map<String, OnnxTensor> decoderInputs = new HashMap<>();
          decoderInputs.put("memory", memory);
          decoderInputs.put("target_tokens", targetTokensTensor);
          decoderInputs.put("target_mask", targetMaskTensor);
          decoderInputs.put("src_mask", srcMaskTensor);
          
          OrtSession.Result decoderOutput = _decoderSession.run(decoderInputs);
          OnnxTensor logitsTensor = (OnnxTensor) decoderOutput.get(0); // Get first output
          
          // Handle 3D logits tensor [1, seq_len, vocab_size]
          long[] logitsShape = logitsTensor.getInfo().getShape(); 
          logDebug("   Logits shape: " + java.util.Arrays.toString(logitsShape));
          
          // Get logits tensor data - debug the actual type first
          Object logitsValue = logitsTensor.getValue();
          logDebug("   Logits value type: " + logitsValue.getClass().getName());
          
          // Extract logits for the position after last real token
          int tokenPosition = Math.min(beam.tokens.size() - 1, decoderSeqLength - 1);
          float[] relevantLogits = new float[vocabSize];
          
          try {
            // Try 3D tensor first [batch, seq_len, vocab_size]
            float[][][] logits3D = (float[][][]) logitsValue;
            logDebug("   Accessing logits3D[0][" + tokenPosition + "] (vocab=" + vocabSize + ")");
            
            // Verify bounds
            if (tokenPosition >= 0 && tokenPosition < logits3D[0].length && 
                logits3D[0][tokenPosition].length >= vocabSize) {
              System.arraycopy(logits3D[0][tokenPosition], 0, relevantLogits, 0, vocabSize);
              logDebug("   ✅ Extracted 3D logits successfully");
            } else {
              throw new RuntimeException("Logits bounds error: pos=" + tokenPosition + 
                ", seq_len=" + logits3D[0].length + 
                ", vocab=" + (logits3D[0].length > 0 ? logits3D[0][0].length : "N/A"));
            }
          }
          catch (ClassCastException e3D) {
            logDebug("   3D cast failed, trying flattened array...");
            try {
              // Try flattened array
              float[] logitsFlat = (float[]) logitsValue;
              int startIdx = tokenPosition * vocabSize;
              logDebug("   Flattened array length: " + logitsFlat.length + ", extracting from " + startIdx);
              
              if (startIdx + vocabSize <= logitsFlat.length) {
                System.arraycopy(logitsFlat, startIdx, relevantLogits, 0, vocabSize);
                logDebug("   ✅ Extracted flattened logits successfully");
              } else {
                throw new RuntimeException("Flattened logits bounds error");
              }
            }
            catch (ClassCastException eFlat) {
              logDebug("   💥 Both 3D and flat casting failed!");
              throw new RuntimeException("Unsupported logits tensor format: " + logitsValue.getClass().getName() + 
                ". Expected float[][][] or float[]");
            }
          }
          
          // Apply softmax
          float[] probs = softmax(relevantLogits);
          
          // Get top k tokens
          int[] topK = getTopKIndices(probs, beamWidth);
          
          for (int idx : topK)
          {
            BeamSearchState newBeam = new BeamSearchState(beam);
            newBeam.tokens.add((long)idx);
            newBeam.score += Math.log(probs[idx]);
            newBeam.finished = (idx == EOS_IDX);
            candidates.add(newBeam);
          }
          
          // Clean up local tensors (don't close srcMaskTensor - managed by caller)
          targetTokensTensor.close();
          targetMaskTensor.close();
          decoderOutput.close();
          
        }
        catch (Exception e)
        {
          logDebug("💥 Decoder step error: " + e.getClass().getSimpleName() + " - " + e.getMessage());
          Log.e(TAG, "Decoder step error", e);
          continue;
        }
      }
      
      // Select top beams
      candidates.sort((a, b) -> Float.compare(b.score, a.score));
      beams = candidates.subList(0, Math.min(candidates.size(), beamWidth));
      
      // Check if all beams finished or have enough predictions
      boolean allFinished = true;
      int finishedCount = 0;
      for (BeamSearchState beam : beams) {
        if (!beam.finished) allFinished = false;
        if (beam.finished) finishedCount++;
      }
      
      if (allFinished || (step >= 10 && finishedCount >= 3))
      {
        break;
      }
    }
    
    // Convert token sequences to words
    List<BeamSearchCandidate> results = new ArrayList<>();
    for (BeamSearchState beam : beams)
    {
      StringBuilder word = new StringBuilder();
      for (Long token : beam.tokens)
      {
        int idx = token.intValue();
        if (idx == SOS_IDX || idx == EOS_IDX || idx == PAD_IDX) continue;
        
        char ch = _tokenizer.indexToChar(idx);
        if (ch != '?' && !Character.toString(ch).startsWith("<"))
        {
          word.append(ch);
        }
      }
      
      String wordStr = word.toString();
      if (wordStr.length() > 0)
      {
        results.add(new BeamSearchCandidate(wordStr, (float)Math.exp(beam.score)));
      }
    }
    
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
  
  private int[] getTopKIndices(float[] array, int k)
  {
    List<IndexValue> indexed = new ArrayList<>();
    for (int i = 0; i < array.length; i++)
    {
      indexed.add(new IndexValue(i, array[i]));
    }
    
    indexed.sort((a, b) -> Float.compare(b.value, a.value));
    
    int[] result = new int[Math.min(k, indexed.size())];
    for (int i = 0; i < result.length; i++) {
      result[i] = indexed.get(i).index;
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
  
  private PredictionResult createPredictionResult(List<BeamSearchCandidate> candidates)
  {
    List<String> words = new ArrayList<>();
    List<Integer> scores = new ArrayList<>();
    
    // Filter candidates through vocabulary system like web demo
    List<String> candidateWords = new ArrayList<>();
    for (BeamSearchCandidate candidate : candidates)
    {
      if (candidate.confidence >= _confidenceThreshold)
      {
        candidateWords.add(candidate.word);
      }
    }
    
    // Apply vocabulary filtering for valid words only
    List<String> filteredWords = _vocabulary.filterPredictions(candidateWords);
    
    // Rebuild scores for filtered words
    for (String word : filteredWords)
    {
      // Find original candidate and score
      for (BeamSearchCandidate candidate : candidates)
      {
        if (candidate.word.equals(word))
        {
          words.add(word);
          scores.add((int)(candidate.confidence * 1000)); // Convert to 0-1000 range
          break;
        }
      }
    }
    
    logDebug("📊 Filtered predictions: " + candidateWords.size() + " → " + filteredWords.size() + " valid words");
    
    return new PredictionResult(words, scores);
  }
  
  private PredictionResult createEmptyResult()
  {
    return new PredictionResult(new ArrayList<>(), new ArrayList<>());
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