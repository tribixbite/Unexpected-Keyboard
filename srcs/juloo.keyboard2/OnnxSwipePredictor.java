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
  
  // Model configuration matching web demo
  private static final int MAX_SEQUENCE_LENGTH = 100;
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
  
  // Model state
  private boolean _isModelLoaded = false;
  private boolean _isInitialized = false;
  
  // Configuration parameters
  private int _beamWidth = DEFAULT_BEAM_WIDTH;
  private int _maxLength = DEFAULT_MAX_LENGTH;
  private float _confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD;
  
  public OnnxSwipePredictor(Context context)
  {
    _context = context;
    _ortEnvironment = OrtEnvironment.getEnvironment();
    _trajectoryProcessor = new SwipeTrajectoryProcessor();
    _tokenizer = new SwipeTokenizer();
    
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
      
      // Load encoder model
      byte[] encoderModelData = loadModelFromAssets("models/swipe_encoder.onnx");
      if (encoderModelData != null)
      {
        _encoderSession = _ortEnvironment.createSession(encoderModelData);
        Log.d(TAG, "Encoder model loaded successfully");
      }
      
      // Load decoder model
      byte[] decoderModelData = loadModelFromAssets("models/swipe_decoder.onnx");
      if (decoderModelData != null)
      {
        _decoderSession = _ortEnvironment.createSession(decoderModelData);
        Log.d(TAG, "Decoder model loaded successfully");
      }
      
      // Load tokenizer configuration
      _tokenizer.loadFromAssets(_context);
      
      _isModelLoaded = (_encoderSession != null && _decoderSession != null);
      _isInitialized = true;
      
      if (_isModelLoaded)
      {
        Log.d(TAG, "ONNX neural prediction system ready");
      }
      else
      {
        Log.w(TAG, "Failed to load ONNX models - will use fallback");
      }
      
      return _isModelLoaded;
    }
    catch (Exception e)
    {
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
        
        Map<String, OnnxTensor> encoderInputs = new HashMap<>();
        encoderInputs.put("trajectory_features", trajectoryTensor);
        encoderInputs.put("nearest_keys", nearestKeysTensor);
        encoderInputs.put("src_mask", srcMaskTensor);
        
        encoderResult = _encoderSession.run(encoderInputs);
      
        // Run beam search decoding
        List<BeamSearchCandidate> candidates = runBeamSearch(encoderResult, features);
        
        // Convert to prediction result
        return createPredictionResult(candidates);
        
      } finally {
        // Proper memory cleanup
        if (trajectoryTensor != null) trajectoryTensor.close();
        if (nearestKeysTensor != null) nearestKeysTensor.close();
        if (srcMaskTensor != null) srcMaskTensor.close();
        if (encoderResult != null) encoderResult.close();
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
      InputStream inputStream = _context.getAssets().open(modelPath);
      byte[] modelData = new byte[inputStream.available()];
      inputStream.read(modelData);
      inputStream.close();
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
    float[] trajectoryData = new float[MAX_SEQUENCE_LENGTH * TRAJECTORY_FEATURES];
    
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
    {
      int baseIdx = i * TRAJECTORY_FEATURES;
      
      if (i < features.normalizedPoints.size())
      {
        SwipeTrajectoryProcessor.TrajectoryPoint point = features.normalizedPoints.get(i);
        trajectoryData[baseIdx + 0] = point.x;
        trajectoryData[baseIdx + 1] = point.y;
        trajectoryData[baseIdx + 2] = point.vx;
        trajectoryData[baseIdx + 3] = point.vy;
        trajectoryData[baseIdx + 4] = point.ax;
        trajectoryData[baseIdx + 5] = point.ay;
      }
      // Padding handled by zero initialization
    }
    
    long[] shape = {1, MAX_SEQUENCE_LENGTH, TRAJECTORY_FEATURES};
    return OnnxTensor.createTensor(_ortEnvironment, java.nio.FloatBuffer.wrap(trajectoryData), shape);
  }
  
  private OnnxTensor createNearestKeysTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    long[] nearestKeysData = new long[MAX_SEQUENCE_LENGTH];
    
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
    {
      if (i < features.nearestKeys.size())
      {
        char key = features.nearestKeys.get(i);
        nearestKeysData[i] = _tokenizer.charToIndex(key);
      }
      else
      {
        nearestKeysData[i] = PAD_IDX; // Padding
      }
    }
    
    long[] shape = {1, MAX_SEQUENCE_LENGTH};
    return OnnxTensor.createTensor(_ortEnvironment, java.nio.LongBuffer.wrap(nearestKeysData), shape);
  }
  
  private OnnxTensor createSourceMaskTensor(SwipeTrajectoryProcessor.TrajectoryFeatures features)
    throws OrtException
  {
    boolean[] srcMaskData = new boolean[MAX_SEQUENCE_LENGTH];
    
    // Mask padded positions (true = masked/padded)
    for (int i = 0; i < MAX_SEQUENCE_LENGTH; i++)
    {
      srcMaskData[i] = (i >= features.actualLength);
    }
    
    long[] shape = {1, MAX_SEQUENCE_LENGTH};
    return OnnxTensor.createTensor(_ortEnvironment, srcMaskData);
  }
  
  private List<BeamSearchCandidate> runBeamSearch(OrtSession.Result encoderResult, 
    SwipeTrajectoryProcessor.TrajectoryFeatures features) throws OrtException
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
          
          // Create target mask - 1 for PADDED positions, 0 for real tokens
          boolean[] tgtMask = new boolean[decoderSeqLength];
          for (int i = beam.tokens.size(); i < decoderSeqLength; i++)
          {
            tgtMask[i] = true; // Mark padded positions
          }
          
          // Create source mask (0 for all positions - all encoder positions valid)
          boolean[] srcMask = new boolean[(int)memory.getInfo().getShape()[1]];
          // All false (no masking) - arrays initialize to false
          
          // Create tensors with proper ONNX API
          OnnxTensor targetTokensTensor = OnnxTensor.createTensor(_ortEnvironment, 
            java.nio.LongBuffer.wrap(paddedTokens), new long[]{1, decoderSeqLength});
          OnnxTensor targetMaskTensor = OnnxTensor.createTensor(_ortEnvironment, tgtMask);
          OnnxTensor srcMaskTensor = OnnxTensor.createTensor(_ortEnvironment, srcMask);
          
          // Run decoder
          Map<String, OnnxTensor> decoderInputs = new HashMap<>();
          decoderInputs.put("memory", memory);
          decoderInputs.put("target_tokens", targetTokensTensor);
          decoderInputs.put("target_mask", targetMaskTensor);
          decoderInputs.put("src_mask", srcMaskTensor);
          
          OrtSession.Result decoderOutput = _decoderSession.run(decoderInputs);
          OnnxTensor logitsTensor = (OnnxTensor) decoderOutput.get(0); // Get first output
          
          // Get logits for position after last real token
          int tokenPosition = Math.min(beam.tokens.size() - 1, decoderSeqLength - 1);
          float[] logitsData = (float[]) logitsTensor.getValue();
          
          int startIdx = tokenPosition * vocabSize;
          float[] relevantLogits = java.util.Arrays.copyOfRange(logitsData, startIdx, startIdx + vocabSize);
          
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
          
          // Clean up tensors
          targetTokensTensor.close();
          targetMaskTensor.close();
          srcMaskTensor.close();
          decoderOutput.close();
          
        }
        catch (Exception e)
        {
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
    
    for (BeamSearchCandidate candidate : candidates)
    {
      if (candidate.confidence >= _confidenceThreshold)
      {
        words.add(candidate.word);
        scores.add((int)(candidate.confidence * 1000)); // Convert to 0-1000 range
      }
    }
    
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