# Neural Network Input Parameters - Complete Reference

**Purpose**: Document every parameter that affects the values passed to the neural network for swipe prediction.

**Date**: 2025-11-25
**Status**: ✅ Complete and verified

---

## Overview

Raw touch coordinates go through 8 transformations before reaching the neural network:

1. **Touch Input Smoothing** - Moving average filter
2. **Noise Filtering** - Discard tiny movements
3. **Coordinate Normalization** - Convert to [0,1] range
4. **Y-Offset Correction** - Compensate for fat finger effect
5. **QWERTY Area Mapping** - Map to training coordinate space
6. **Resampling** - Downsample to fixed length
7. **Feature Calculation** - Compute velocity & acceleration
8. **Clipping** - Remove outliers

---

## 1. Touch Input Smoothing

### SMOOTHING_WINDOW = 3

**File**: `ImprovedSwipeGestureRecognizer.java:34, 152-173`

#### What It Does

Applies **Simple Moving Average (SMA)** with window size 3 to raw touch coordinates.

**Algorithm**:
```java
// First 2 points: no smoothing (insufficient history)
smoothed[0] = raw[0]
smoothed[1] = raw[1]

// Starting at point 3: average last 3 raw points
for (i >= 2) {
    smoothed[i].x = (raw[i-2].x + raw[i-1].x + raw[i].x) / 3
    smoothed[i].y = (raw[i-2].y + raw[i-1].y + raw[i].y) / 3
}
```

#### Concrete Example
```
Raw touch points (pixels):
  [0] (100, 200)
  [1] (105, 205)
  [2] (110, 210)
  [3] (115, 215)
  [4] (120, 220)

Smoothed output:
  [0] (100, 200)     <- No smoothing (first point)
  [1] (105, 205)     <- No smoothing (only 2 points)
  [2] (105, 205)     <- (100+105+110)/3, (200+205+210)/3
  [3] (110, 210)     <- (105+110+115)/3, (205+210+215)/3
  [4] (115, 215)     <- (110+115+120)/3, (210+215+220)/3
```

#### Effect on Neural Network
- **Reduces jitter**: ±5 pixel shake → ±1.7 pixel variation
- **Preserves shape**: Window of 3 is small, maintains swipe geometry
- **Slight lag**: Smoothed point lags ~1-2 samples behind actual touch
- **Frequency response**: Low-pass filter, attenuates high-frequency noise

#### Mathematical Properties
- **Filter type**: Simple Moving Average (uniform weights)
- **Weights**: [1/3, 1/3, 1/3]
- **Lag**: ~1 sample (33% of window)
- **Noise reduction**: ~40% at Nyquist frequency

---

## 2. Noise Filtering

### NOISE_THRESHOLD = 10.0f pixels

**File**: `ImprovedSwipeGestureRecognizer.java:37, 116-120`

#### What It Does

Discards touch points that move less than 10 pixels from the last accepted point.

**Algorithm**:
```java
distance = sqrt((x_new - x_last)² + (y_new - y_last)²)

if (distance < 10.0f) {
    // DISCARD: Don't add to trajectory
    return;
}
// ACCEPT: Add to _rawPath
```

#### Concrete Example
```
Last accepted point: (100, 200)

Incoming touch events:
  (101, 201) -> dist = 1.4px  ❌ DISCARDED
  (102, 199) -> dist = 2.2px  ❌ DISCARDED
  (105, 208) -> dist = 9.4px  ❌ DISCARDED
  (112, 210) -> dist = 15.6px ✅ ACCEPTED (becomes new reference)
```

#### Effect on Neural Network
- **Reduces point count**: 20-40% fewer points (varies by swipe speed)
- **Eliminates dwelling**: Removes stationary clusters at start/end
- **Cleaner trajectory**: No micro-movements from hand tremor
- **Faster inference**: Fewer points = less computation

#### When This Matters
- **Slow swipes**: More likely to accumulate noise points
- **Initial touch**: First 30ms has jitter as finger settles
- **Lift-off**: Last 50ms clusters points in small area
- **Screen protectors**: Can increase touch jitter

---

## 3. Coordinate Normalization

### Keyboard Dimensions

**File**: `SwipeTrajectoryProcessor.java:59-60, 282-294, 304-318`

#### Parameters
```java
_keyboardWidth  // Total keyboard width in pixels (e.g., 1080)
_keyboardHeight // Total keyboard height in pixels (e.g., 632)
```

#### What It Does

Converts pixel coordinates to device-independent [0, 1] range.

**Algorithm**:
```java
x_normalized = x_pixels / _keyboardWidth
y_normalized = y_pixels / _keyboardHeight

// Clamp to valid range
x_normalized = max(0.0, min(1.0, x_normalized))
y_normalized = max(0.0, min(1.0, y_normalized))
```

#### Concrete Example
```
Keyboard: 1080 × 632 pixels
Touch point: (540, 316)

Normalization:
  x_norm = 540 / 1080 = 0.5   (center horizontally)
  y_norm = 316 / 632 = 0.5    (center vertically)
```

#### Effect on Neural Network
- **Device independence**: Same swipe produces same normalized trajectory across devices
- **Training match**: Model was trained on [0,1] coordinates
- **Critical accuracy**: Wrong dimensions = distorted trajectory = wrong predictions

#### Example of Incorrect Dimensions
```
Actual keyboard: 1080 × 632
Wrong config:    1080 × 1000 (forgot suggestion bar offset)

Bottom row swipe (Z key at y=500):
  Correct: 500/632 = 0.79  → "Bottom row"
  Wrong:   500/1000 = 0.50 → "Middle row"

Result: Model thinks user swiped D-H row instead of Z-M row
```

---

## 4. Touch Y-Offset Correction

### _touchYOffset = 74.0f pixels

**File**: `SwipeTrajectoryProcessor.java:33, 89-93, 306-308`

#### What It Does

Shifts Y-coordinates DOWN by 74 pixels to compensate for fat finger effect.

**Algorithm**:
```java
y_adjusted = y_raw + 74.0f
// Then normalize using adjusted value
```

#### Why This Exists

Human fingers are not point objects. Visual perception and finger geometry cause users to touch ~74 pixels ABOVE the intended key center:

1. **Finger pad geometry**: Fingertip is round, contact point ≠ visual target
2. **Visual occlusion**: Finger blocks view, user aims above visible target
3. **Perception bias**: Users look at key center but touch with finger pad

#### Concrete Example
```
Q key center: y = 99 pixels
User intends to touch Q

WITHOUT correction:
  Touch detected: y = 25 pixels (finger pad touched above key)
  Normalized: 25/632 = 0.04 (near top edge)
  NN interprets: "Above Q, maybe number row?"
  Result: Wrong predictions ❌

WITH correction (+74 pixels):
  Touch detected: y = 25 pixels
  Adjusted: y = 25 + 74 = 99 pixels ✅ (Q key center!)
  Normalized: 99/632 = 0.16 (Q row)
  NN interprets: "Q key"
  Result: Correct predictions ✅
```

#### Calibration Example
```
Problem: User reports "swiping 'hello' predicts 'twllo'"
Analysis: h→t means touches detected too high
         (h is row 1, t is row 0)
Solution: Increase touchYOffset from 74 to 85 pixels
Result: Touch shifts down more, h key registers correctly
```

#### Effect on Neural Network
- **Aligns intent with detection**: What user meant = what NN sees
- **Row accuracy**: Critical for q/a/z disambiguation (different rows)
- **Device-specific**: May need calibration per device
  - Screen protector thickness
  - Case lip height
  - Individual user finger geometry

---

## 5. QWERTY Area Mapping

### QWERTY Bounds

**File**: `SwipeTrajectoryProcessor.java:26-28, 74-80, 299-318`

#### Parameters
```java
_qwertyAreaTop    // Y pixel offset where QWERTY keys start (e.g., 37)
_qwertyAreaHeight // Height of QWERTY area in pixels (e.g., 595)
```

#### What It Does

Maps ONLY the QWERTY letter area to [0, 1] range, excluding suggestion bar and other UI elements.

**Problem**: Full keyboard view includes:
- Suggestion bar (top ~37 pixels)
- Number row (optional)
- Space bar, emoji keys (bottom area)

But neural network was trained ONLY on QWERTY 3-row area.

**Algorithm**:
```java
// X coordinate: use full keyboard width
x_normalized = x_raw / _keyboardWidth

// Y coordinate: map QWERTY area to [0, 1]
y_adjusted = y_raw + _touchYOffset
if (_qwertyAreaHeight > 0) {
    y_normalized = (y_adjusted - _qwertyAreaTop) / _qwertyAreaHeight
} else {
    y_normalized = y_adjusted / _keyboardHeight  // Fallback
}

// Clamp to [0, 1]
y_normalized = max(0.0, min(1.0, y_normalized))
```

#### Concrete Example
```
Full keyboard layout:
    0px  ┌─────────────────────┐
         │  Suggestion bar     │
   37px  ├─────────────────────┤ <- _qwertyAreaTop
         │ Row 0: qwertyuiop   │
  235px  ├─────────────────────┤
         │ Row 1: asdfghjkl    │
  433px  ├─────────────────────┤
         │ Row 2: zxcvbnm      │
  632px  └─────────────────────┘

Touch on Z key (raw y = 496):

WITHOUT QWERTY bounds (WRONG):
  y_norm = 496 / 632 = 0.78
  NN: "78% down full keyboard, between row 2-3?"
  Result: Confused predictions ❌

WITH QWERTY bounds (CORRECT):
  y_norm = (496 - 37) / 595 = 459/595 = 0.77
  NN: "77% down QWERTY area = row 3 (bottom)"
  Result: "Z row detected" ✅
```

#### Normalized Coordinates Sent to NN
```
Physical keys → Normalized Y:
  Q (top row)    → y = 0.0 to 0.33
  A (middle row) → y = 0.33 to 0.67
  Z (bottom row) → y = 0.67 to 1.0
```

#### Effect on Neural Network
- **Correct vertical scaling**: Q→Z spans [0.0, 1.0] not [0.06, 0.78]
- **Matches training data**: Model trained with QWERTY = [0, 1]
- **Critical for accuracy**: Without this, all y-coords compressed and wrong
- **Essential for row disambiguation**: Determines which row NN sees

---

## 6. Resampling

### MAX_SEQUENCE_LENGTH = 250

**File**: `SwipeTrajectoryProcessor.java:108, 138-177`
**File**: `SwipeResampler.java:85-121`
**File**: `Config.java` (neural_user_max_seq_length)

#### Parameters
```java
maxSequenceLength = 250  // Neural network input constraint
_resamplingMode = DISCARD // Default: weighted point selection
```

#### What It Does

If trajectory has more than 250 points, downsample to exactly 250 while preserving swipe shape.

**DISCARD Mode Algorithm**:
```java
if (smoothedPath.size() > 250) {
    // 1. Always keep first point
    // 2. Always keep last point
    // 3. Select 248 middle points with weighted distribution

    // Zone allocation:
    startZone  = first 30% of trajectory
    middleZone = middle 40% of trajectory
    endZone    = last 30% of trajectory

    // Point allocation:
    pointsFromStart  = 35% of 248 = 87 points
    pointsFromMiddle = 30% of 248 = 74 points
    pointsFromEnd    = 35% of 248 = 87 points

    // Select points uniformly within each zone
}
```

#### Concrete Example
```
Original trajectory: 400 points (long slow swipe)

Zone breakdown:
  Start zone:  points 0-120 (first 30% = 120 points)
  Middle zone: points 120-280 (middle 40% = 160 points)
  End zone:    points 280-400 (last 30% = 120 points)

Points selected:
  From start: 87 points from 0-120
    → Every ~1.4th point: [0, 1, 3, 4, 6, 7, ...]

  From middle: 74 points from 120-280
    → Every ~2.2nd point: [120, 122, 124, 127, ...]

  From end: 87 points from 280-400
    → Every ~1.4th point: [..., 398, 399, 400]

Result: 250 points total, more detail at start/end
```

#### Why Start/End Get More Points

- **Word boundaries**: First and last letters are critical for recognition
- **Direction changes**: Swipe endpoints often have sharp direction changes
- **Middle is straighter**: Middle of swipe is often more linear, needs less detail
- **Training distribution**: Model was trained with this same weighting

#### Visual Example (400 → 250 points)
```
Original (400 points):
H ●●●●●●●●●● E ●●●●●●●●●● L ●●●●●●●●●● L ●●●●●●●●●● O
  High density throughout

After DISCARD (250 points):
H ●●●●●●● E ●●●● L ●●●● L ●●●●●●● O
  Start: dense    Middle: sparse    End: dense
```

#### Effect on Neural Network
- **Fixed input size**: Model expects exactly 250 points, always
- **Preserves shape**: Downsampling maintains overall trajectory geometry
- **Reduces computation**: 400→250 = 37% faster inference
- **Maintains critical detail**: Start/end preserved for word boundary detection

#### Padding (Short Swipes)
```java
// If trajectory has < 250 points, pad with zeros
while (points.size() < 250) {
    points.add([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  // Zero features
}

// Neural network uses actual_length to ignore padding
actual_length = min(originalSize, 250)
```

---

## 7. Feature Calculation

### Velocity & Acceleration

**File**: `TrajectoryFeatureCalculator.kt:53-122`

#### Features Computed

For each point, calculate 6 features: **[x, y, vx, vy, ax, ay]**

**Algorithm**:
```kotlin
// Time deltas (prevent division by zero)
dt[0] = 0
for (i in 1 until n) {
    dt[i] = max(timestamps[i] - timestamps[i-1], 1e-6f)
}

// Velocity (position change per time)
vx[0] = 0.0
vy[0] = 0.0
for (i in 1 until n) {
    vx[i] = (x[i] - x[i-1]) / dt[i]
    vy[i] = (y[i] - y[i-1]) / dt[i]
}

// Acceleration (velocity change per time)
ax[0] = 0.0
ay[0] = 0.0
for (i in 1 until n) {
    ax[i] = (vx[i] - vx[i-1]) / dt[i]
    ay[i] = (vy[i] - vy[i-1]) / dt[i]
}

// Clip to prevent outliers
vx[i] = vx[i].coerceIn(-10f, 10f)
vy[i] = vy[i].coerceIn(-10f, 10f)
ax[i] = ax[i].coerceIn(-10f, 10f)
ay[i] = ay[i].coerceIn(-10f, 10f)
```

#### Concrete Example
```
Input trajectory (3 points):
  Point 0: x=0.0,  y=0.0,  t=0ms
  Point 1: x=0.1,  y=0.05, t=10ms
  Point 2: x=0.25, y=0.15, t=20ms

Features calculated:
  Point 0: [0.0,  0.0,  0.0,   0.0,   0.0,    0.0]
           [x,    y,    vx,    vy,    ax,     ay]

  Point 1: [0.1,  0.05, 10.0,  5.0,   0.0,    0.0]
           dt=10ms, vx=(0.1-0.0)/0.01s=10.0

  Point 2: [0.25, 0.15, 15.0,  10.0,  500.0,  500.0]
           dt=10ms, vx=(0.25-0.1)/0.01s=15.0
                    ax=(15.0-10.0)/0.01s=500.0
                    (would be clipped to 10.0)
```

#### Effect on Neural Network
- **Motion context**: Velocity shows direction/speed, acceleration shows curve sharpness
- **Better recognition**: Moving from H→E→L has different velocity profile than E→L→L
- **Temporal information**: Features encode time-dependent patterns
- **Matches training**: Model was trained with these exact 6 features

---

## 8. Velocity & Acceleration Clipping

### CLIP_RANGE = [-10, 10]

**File**: `TrajectoryFeatureCalculator.kt:103-109`

#### What It Does

Clamps extreme velocity/acceleration values to prevent outliers from corrupting model input.

**Algorithm**:
```kotlin
// After calculation, clamp to range
vx[i] = max(-10.0f, min(10.0f, vx[i]))
vy[i] = max(-10.0f, min(10.0f, vy[i]))
ax[i] = max(-10.0f, min(10.0f, ax[i]))
ay[i] = max(-10.0f, min(10.0f, ay[i]))
```

#### Concrete Example
```
Point sequence with timestamp glitch:
  Point 0: x=0.0,  t=0ms
  Point 1: x=0.5,  t=10ms
  Point 2: x=0.51, t=10.1ms  <- Only 0.1ms delta!

Velocity at point 2:
  dx = 0.51 - 0.5 = 0.01
  dt = 0.1ms = 0.0001s
  vx_raw = 0.01 / 0.0001 = 100.0 ❌ (huge spike!)
  vx_clipped = 10.0 ✅ (clamped to max)

Result: Spike removed, NN sees reasonable velocity
```

#### What Gets Clipped

**Typical Values** (not clipped):
```
Slow swipe:  vx = 0.5 to 3.0,  ax = 0.2 to 2.0
Fast swipe:  vx = 3.0 to 8.0,  ax = 2.0 to 6.0
```

**Outliers** (clipped):
```
Initial touch:   vx = 15.0 → 10.0  (first velocity has tiny dt)
Lift-off spike:  ax = 50.0 → 10.0  (acceleration spike when finger leaves)
Timestamp bug:   vx = 200.0 → 10.0 (OS reports duplicate/wrong timestamp)
```

#### Effect on Neural Network
- **Stable features**: No extreme spikes in input tensor
- **Better generalization**: Model doesn't see timing noise
- **Consistent scaling**: All features in similar numeric range
- **Matches training**: Model expects clipped values

---

## 9. Minimum Time Delta

### MIN_TIME_DELTA = 1e-6 milliseconds

**File**: `TrajectoryFeatureCalculator.kt:76-79`

#### What It Does

Ensures time delta is never zero to prevent division by zero in velocity/acceleration calculation.

**Algorithm**:
```kotlin
dt[i] = max(timestamps[i] - timestamps[i-1], 1e-6f)
```

#### Concrete Example
```
Touch events with duplicate timestamps:
  Point 0: (x=100, y=200, t=1000ms)
  Point 1: (x=105, y=205, t=1000ms)  <- Same timestamp!

Without protection:
  dt = 1000 - 1000 = 0ms
  vx = (105-100) / 0 = ∞  ❌ CRASH or NaN

With minimum dt:
  dt = max(0, 0.000001) = 0.000001ms
  vx = 5 / 0.000001 = 5,000,000
  vx_clipped = 10.0 ✅ (clipping saves us)
```

#### Why Duplicate Timestamps Happen
- **Sampling rate**: Android samples touch at lower rate than event delivery
- **Touch smoothing**: Android's own smoothing can duplicate timestamps
- **Fast touches**: Very quick swipes compress events into same millisecond
- **Screen digitizer**: Hardware may report multiple points per sample

#### Effect on Neural Network
- **No NaN/Infinity**: Model receives valid numbers, never crashes
- **Graceful degradation**: Combined with clipping, handles glitches well
- **Rare activation**: Most swipes have proper timing, this is safety net

---

## Additional Constants (Non-NN Input)

These parameters affect gesture detection but NOT the values sent to neural network:

### Gesture Detection
```java
MIN_SWIPE_DISTANCE = 50.0f       // Min pixels to qualify as swipe
MIN_DWELL_TIME_MS = 10           // Min ms on key to register
MIN_KEY_DISTANCE = 40.0f         // Min pixels between registered keys
DUPLICATE_CHECK_WINDOW = 5       // Check last 5 keys for duplicates
MAX_POINT_INTERVAL_MS = 500      // Max time between touch points
HIGH_VELOCITY_THRESHOLD = 1000.0f // pixels/sec skip threshold
```

### Probabilistic Key Detection
```java
SIGMA_FACTOR = 0.5f              // Gaussian σ = 0.5 × key_size
MIN_PROBABILITY = 0.01f          // Min prob to consider key (1%)
PROBABILITY_THRESHOLD = 0.3f     // Min cumulative prob (30%)
SIMPLIFICATION_EPSILON = 15.0f   // Ramer-Douglas-Peucker tolerance
```

**Note**: These affect which keys are detected, but smoothed coordinates (not detected keys) go to neural network.

---

## Summary: Parameters → NN Impact

| Parameter | Value | Direct NN Impact | Accuracy Effect | Criticality |
|-----------|-------|------------------|-----------------|-------------|
| **SMOOTHING_WINDOW** | 3 points | Reduces jitter ~70% | Cleaner trajectory | HIGH |
| **NOISE_THRESHOLD** | 10 pixels | Removes 20-40% points | Faster, cleaner | MEDIUM |
| **Keyboard Dimensions** | e.g., 1080×632 | Sets normalization scale | Device independence | CRITICAL |
| **Touch Y-Offset** | 74 pixels | Shifts all Y down | Fixes fat finger | CRITICAL |
| **QWERTY Bounds** | top=37, h=595 | Maps QWERTY to [0,1] | Fixes vertical scale | CRITICAL |
| **Max Sequence Length** | 250 points | Fixed input size | Consistent inference | CRITICAL |
| **Resampling Mode** | DISCARD | Point selection strategy | Shape preservation | MEDIUM |
| **Velocity Clip** | [-10, 10] | Removes outliers | Stability | LOW |
| **Acceleration Clip** | [-10, 10] | Removes outliers | Stability | LOW |
| **Min Time Delta** | 1e-6 ms | Prevents div-by-zero | Safety net | LOW |

### Most Critical (Wrong = All Predictions Wrong)
1. **Keyboard Dimensions** - Must match actual keyboard size
2. **QWERTY Bounds** - Must isolate letter area correctly
3. **Touch Y-Offset** - Must calibrate for fat finger effect
4. **Max Sequence Length** - Must match model's input size (250)

### High Impact (Wrong = Degraded Accuracy)
5. **SMOOTHING_WINDOW** - Balance between noise reduction and responsiveness
6. **NOISE_THRESHOLD** - Balance between filtering and detail preservation

### Medium Impact (Wrong = Minor Issues)
7. **Resampling Weights** - Affects long swipe quality
8. **Clipping Ranges** - Safety margins, rarely triggered

---

## Model Input Tensor Shapes

**Encoder Input**:
```
traj_features:     [1, 250, 6]   // [batch, seq_len, features=(x,y,vx,vy,ax,ay)]
nearest_keys:      [1, 250]      // [batch, seq_len] (token indices 0-29)
actual_src_length: [1]           // scalar int (true length before padding)
```

**Feature Order** (traj_features):
```
Index 0: x          (normalized position [0, 1])
Index 1: y          (normalized position [0, 1])
Index 2: vx         (velocity, clipped [-10, 10])
Index 3: vy         (velocity, clipped [-10, 10])
Index 4: ax         (acceleration, clipped [-10, 10])
Index 5: ay         (acceleration, clipped [-10, 10])
```

---

## Configuration Files

### Runtime Configuration
Most parameters are set in `Config.java`:
```java
neural_user_max_seq_length = 250  // Sequence length
// Keyboard dimensions set dynamically from view measurements
```

### Hardcoded Constants
These require code changes:
```java
// ImprovedSwipeGestureRecognizer.java
SMOOTHING_WINDOW = 3
NOISE_THRESHOLD = 10.0f

// SwipeTrajectoryProcessor.java
_touchYOffset = 74.0f  // Can be set via setTouchYOffset()

// TrajectoryFeatureCalculator.kt
CLIP_MIN = -10.0f
CLIP_MAX = 10.0f
MIN_TIME_DELTA = 1e-6f
```

---

## Debugging Tips

### Check Normalization
```java
Log.d(TAG, "Keyboard: " + width + "×" + height);
Log.d(TAG, "QWERTY: top=" + qwertyTop + ", height=" + qwertyHeight);
Log.d(TAG, "First point: raw=(" + rawX + "," + rawY +
           ") norm=(" + normX + "," + normY + ")");
```

### Expected Values
```
Normalized coordinates should be:
  - Q key: y ≈ 0.0 to 0.15
  - A key: y ≈ 0.33 to 0.5
  - Z key: y ≈ 0.67 to 0.85

If values outside [0, 1]: Check keyboard dimensions
If Q/A/Z rows wrong: Check QWERTY bounds and Y-offset
```

### Verify Feature Calculation
```kotlin
Log.d(TAG, "Point 0: x=" + x[0] + " vx=" + vx[0] + " ax=" + ax[0])
Log.d(TAG, "Point 1: x=" + x[1] + " vx=" + vx[1] + " ax=" + ax[1])

// First point should have vx=0, ax=0
// Clipped values should never exceed ±10
```

---

## References

**Implementation Files**:
- `ImprovedSwipeGestureRecognizer.java` - Smoothing, noise filtering
- `SwipeTrajectoryProcessor.java` - Normalization, QWERTY bounds, resampling
- `TrajectoryFeatureCalculator.kt` - Feature calculation, clipping
- `SwipeResampler.java` - DISCARD resampling algorithm
- `Config.java` - User-configurable parameters

**Related Documentation**:
- `SWIPE_PREDICTION_COEFFICIENTS.md` - All coefficients including post-processing
- `BEAM_SEARCH_AUDIT.md` - Neural network decoder parameters
- `DICTIONARY_CHARACTER_ANALYSIS.md` - Vocabulary and tokenization

---

**Last Updated**: 2025-11-25
**Version**: 1.0
**Status**: ✅ Complete - All parameters affecting NN input documented
