package juloo.keyboard2;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for resampling swipe trajectories to fit different model input sizes
 * Supports three resampling modes:
 * - TRUNCATE: Keep first N points (current behavior)
 * - DISCARD: Drop points uniformly with preference for keeping start/end
 * - MERGE: Average neighboring points to reduce count
 */
public class SwipeResampler
{
  private static final String TAG = "SwipeResampler";

  public enum ResamplingMode
  {
    TRUNCATE,  // Keep first N points, discard rest
    DISCARD,   // Uniformly drop points with start/end preference
    MERGE      // Average neighboring points
  }

  /**
   * Resample trajectory data to target length
   *
   * @param trajectoryData Original data [N, features]
   * @param targetLength Desired output length
   * @param mode Resampling algorithm to use
   * @return Resampled data [targetLength, features]
   */
  public static float[][] resample(float[][] trajectoryData, int targetLength, ResamplingMode mode)
  {
    if (trajectoryData == null || trajectoryData.length == 0)
    {
      return trajectoryData;
    }

    int originalLength = trajectoryData.length;
    int numFeatures = trajectoryData[0].length;

    // No resampling needed
    if (originalLength <= targetLength)
    {
      return trajectoryData;
    }

    switch (mode)
    {
      case TRUNCATE:
        return resampleTruncate(trajectoryData, targetLength);
      case DISCARD:
        return resampleDiscard(trajectoryData, targetLength);
      case MERGE:
        return resampleMerge(trajectoryData, targetLength);
      default:
        return resampleTruncate(trajectoryData, targetLength);
    }
  }

  /**
   * TRUNCATE mode: Keep first targetLength points
   */
  private static float[][] resampleTruncate(float[][] data, int targetLength)
  {
    int numFeatures = data[0].length;
    float[][] result = new float[targetLength][numFeatures];

    for (int i = 0; i < targetLength; i++)
    {
      System.arraycopy(data[i], 0, result[i], 0, numFeatures);
    }

    return result;
  }

  /**
   * DISCARD mode: Drop points semi-uniformly with preference for keeping start and end
   *
   * Strategy:
   * - Always keep first and last points
   * - For middle points, use weighted uniform spacing
   * - Weight more points toward start and end (crucial for word recognition)
   */
  private static float[][] resampleDiscard(float[][] data, int targetLength)
  {
    int originalLength = data.length;
    int numFeatures = data[0].length;
    float[][] result = new float[targetLength][numFeatures];

    if (targetLength == 1)
    {
      // Edge case: keep first point
      System.arraycopy(data[0], 0, result[0], 0, numFeatures);
      return result;
    }

    // Always keep first point
    System.arraycopy(data[0], 0, result[0], 0, numFeatures);

    // Always keep last point
    System.arraycopy(data[originalLength - 1], 0, result[targetLength - 1], 0, numFeatures);

    if (targetLength == 2)
    {
      return result;
    }

    // For middle points, use weighted selection
    // Preserve more points at start and end (first/last 20% of swipe)
    int numMiddle = targetLength - 2;
    List<Integer> selectedIndices = selectMiddleIndices(originalLength, numMiddle);

    for (int i = 0; i < numMiddle; i++)
    {
      int sourceIdx = selectedIndices.get(i);
      System.arraycopy(data[sourceIdx], 0, result[i + 1], 0, numFeatures);
    }

    return result;
  }

  /**
   * Select middle indices with weighted preference for start and end
   */
  private static List<Integer> selectMiddleIndices(int originalLength, int numMiddle)
  {
    List<Integer> indices = new ArrayList<>();

    // Available indices (excluding first and last)
    int availableRange = originalLength - 2;

    if (availableRange <= numMiddle)
    {
      // Keep all middle points
      for (int i = 1; i < originalLength - 1; i++)
      {
        indices.add(i);
      }
      return indices;
    }

    // Use weighted selection: more points at start/end
    // Split into 3 zones: start (30%), middle (40%), end (30%)
    int startZoneEnd = 1 + (int)(availableRange * 0.3);
    int endZoneStart = originalLength - 1 - (int)(availableRange * 0.3);

    int pointsInStart = (int)(numMiddle * 0.35);
    int pointsInEnd = (int)(numMiddle * 0.35);
    int pointsInMiddle = numMiddle - pointsInStart - pointsInEnd;

    // Select from start zone
    for (int i = 0; i < pointsInStart; i++)
    {
      int idx = 1 + (i * (startZoneEnd - 1)) / pointsInStart;
      indices.add(idx);
    }

    // Select from middle zone
    int middleZoneSize = endZoneStart - startZoneEnd;
    for (int i = 0; i < pointsInMiddle; i++)
    {
      int idx = startZoneEnd + (i * middleZoneSize) / pointsInMiddle;
      indices.add(idx);
    }

    // Select from end zone
    int endZoneSize = (originalLength - 1) - endZoneStart;
    for (int i = 0; i < pointsInEnd; i++)
    {
      int idx = endZoneStart + (i * endZoneSize) / pointsInEnd;
      indices.add(idx);
    }

    return indices;
  }

  /**
   * MERGE mode: Average neighboring points to reduce count
   *
   * Strategy:
   * - Calculate merge factor (how many original points per output point)
   * - For each output point, average the corresponding range of input points
   * - Preserves overall trajectory shape better than discard
   */
  private static float[][] resampleMerge(float[][] data, int targetLength)
  {
    int originalLength = data.length;
    int numFeatures = data[0].length;
    float[][] result = new float[targetLength][numFeatures];

    // Calculate how many source points map to each target point
    float mergeFactor = (float)originalLength / targetLength;

    for (int targetIdx = 0; targetIdx < targetLength; targetIdx++)
    {
      // Calculate source range for this target point
      float startFloat = targetIdx * mergeFactor;
      float endFloat = (targetIdx + 1) * mergeFactor;

      int startIdx = (int)startFloat;
      int endIdx = (int)Math.ceil(endFloat);
      endIdx = Math.min(endIdx, originalLength);

      // Average all points in this range
      float[] avgPoint = new float[numFeatures];
      int count = 0;

      for (int sourceIdx = startIdx; sourceIdx < endIdx; sourceIdx++)
      {
        for (int f = 0; f < numFeatures; f++)
        {
          avgPoint[f] += data[sourceIdx][f];
        }
        count++;
      }

      // Compute average
      for (int f = 0; f < numFeatures; f++)
      {
        result[targetIdx][f] = avgPoint[f] / count;
      }
    }

    return result;
  }

  /**
   * Parse resampling mode from string
   */
  public static ResamplingMode parseMode(String modeString)
  {
    if (modeString == null)
    {
      return ResamplingMode.TRUNCATE;
    }

    switch (modeString.toLowerCase())
    {
      case "discard":
        return ResamplingMode.DISCARD;
      case "merge":
        return ResamplingMode.MERGE;
      case "truncate":
      default:
        return ResamplingMode.TRUNCATE;
    }
  }
}
