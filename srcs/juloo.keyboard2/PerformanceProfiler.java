package juloo.keyboard2;

import android.os.Trace;
import android.util.Log;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Performance profiler using Android's standard Trace API.
 *
 * OPTIMIZATION v2: Uses android.os.Trace for system-level profiling
 * - Integrates with Perfetto and Android Studio Profiler
 * - Zero overhead in release builds (traces are compiled out)
 * - Provides accurate system-level performance analysis
 *
 * Legacy statistics tracking is kept for development but disabled by default.
 * Enable ENABLE_STATISTICS in debug builds for detailed timing analysis.
 */
public class PerformanceProfiler
{
  private static final String TAG = "PerfProfiler";

  // OPTIMIZATION: Use BuildConfig for compile-time optimization in release builds
  private static final boolean ENABLE_TRACE = BuildConfig.DEBUG; // Android Trace (always enabled in debug)
  private static final boolean ENABLE_STATISTICS = false; // Legacy stats (disabled for performance)

  private static final Map<String, Long> _startTimes = new ConcurrentHashMap<>();
  private static final Map<String, Statistics> _statistics = new ConcurrentHashMap<>();

  /**
   * Statistics for a profiled section (legacy, disabled by default)
   */
  private static class Statistics
  {
    long count = 0;
    long totalTime = 0;
    long minTime = Long.MAX_VALUE;
    long maxTime = 0;

    synchronized void record(long time)
    {
      count++;
      totalTime += time;
      minTime = Math.min(minTime, time);
      maxTime = Math.max(maxTime, time);
    }

    long getAverage()
    {
      return count > 0 ? totalTime / count : 0;
    }
  }

  /**
   * Start timing a section.
   *
   * OPTIMIZATION v2: Uses android.os.Trace.beginSection() for system-level profiling.
   * Integrates with Perfetto, Android Studio Profiler, and systrace.
   *
   * @param section Section name (max 127 characters)
   */
  public static void start(String section)
  {
    // Android Trace API (compiled out in release builds)
    if (ENABLE_TRACE)
    {
      Trace.beginSection(section);
    }

    // Legacy statistics (optional, disabled by default for performance)
    if (ENABLE_STATISTICS)
    {
      _startTimes.put(section, System.currentTimeMillis());
    }
  }

  /**
   * End timing a section.
   *
   * OPTIMIZATION v2: Uses android.os.Trace.endSection() for system-level profiling.
   */
  public static void end(String section)
  {
    // Android Trace API (compiled out in release builds)
    if (ENABLE_TRACE)
    {
      Trace.endSection();
    }

    // Legacy statistics (optional, disabled by default)
    if (ENABLE_STATISTICS)
    {
      Long startTime = _startTimes.remove(section);
      if (startTime == null)
      {
        if (BuildConfig.ENABLE_VERBOSE_LOGGING) {
          Log.w(TAG, "No start time for section: " + section);
        }
        return;
      }

      long duration = System.currentTimeMillis() - startTime;

      Statistics stats = _statistics.computeIfAbsent(section, k -> new Statistics());
      stats.record(duration);

      // Log if this operation took longer than expected
      if (duration > getThreshold(section))
      {
        Log.w(TAG, String.format("%s took %dms (threshold: %dms)",
                                 section, duration, getThreshold(section)));
      }
    }
  }
  
  /**
   * Get performance thresholds for different operations
   */
  private static long getThreshold(String section)
  {
    // Define expected maximum times for operations
    if (section.startsWith("DTW."))
      return 30; // DTW should complete within 30ms
    if (section.startsWith("Swipe."))
      return 50; // Swipe prediction should complete within 50ms
    if (section.startsWith("Type."))
      return 20; // Regular typing prediction within 20ms
    if (section.startsWith("Gaussian."))
      return 10; // Gaussian model within 10ms
    if (section.startsWith("Ngram."))
      return 5; // N-gram lookup within 5ms
    return 100; // Default threshold
  }
  
  /**
   * Print performance report
   */
  public static void report()
  {
    if (!ENABLE_STATISTICS || _statistics.isEmpty()) return;
    
    Log.d(TAG, "===== PERFORMANCE REPORT =====");
    for (Map.Entry<String, Statistics> entry : _statistics.entrySet())
    {
      String section = entry.getKey();
      Statistics stats = entry.getValue();
      
      Log.d(TAG, String.format("%s: count=%d, avg=%dms, min=%dms, max=%dms, total=%dms",
                              section, stats.count, stats.getAverage(),
                              stats.minTime, stats.maxTime, stats.totalTime));
    }
    Log.d(TAG, "==============================");
  }
  
  /**
   * Clear all statistics
   */
  public static void reset()
  {
    _startTimes.clear();
    _statistics.clear();
  }
  
  /**
   * Log a single timing event without tracking statistics
   */
  public static void logTiming(String operation, long timeMs)
  {
    if (!ENABLE_STATISTICS) return;
    
    if (timeMs > getThreshold(operation))
    {
      Log.w(TAG, String.format("%s: %dms (SLOW)", operation, timeMs));
    }
    else
    {
      Log.d(TAG, String.format("%s: %dms", operation, timeMs));
    }
  }
}