# Performance Bottleneck Analysis: Swipe UI Rendering & Input Handling

**Date**: 2025-11-21
**Status**: ⚠️ **New Bottleneck Identified**

## Executive Summary

While the neural inference path has been optimized with object pooling (`SwipeTrajectoryProcessor`), the **UI rendering and input handling path** for swipes remains unoptimized. This causes significant object allocation on the UI thread during every frame of a swipe gesture, leading to GC pressure and potential visual jank.

## Detailed Findings

### 1. Excessive Allocations in `ImprovedSwipeGestureRecognizer`

**File**: `srcs/juloo.keyboard2/ImprovedSwipeGestureRecognizer.java`

**Issue**:
The recognizer allocates multiple `PointF` objects for **every single touch move event**.
*   `startSwipe`: `new PointF(x, y)`
*   `addPoint`: `new PointF(x, y)`
*   `applySmoothing`: `new PointF(avgX / count, avgY / count)`

**Impact**:
Input events occur at a high frequency (60-120Hz+). Allocating 2-3 objects per event generates thousands of short-lived objects during a single swipe, triggering frequent minor Garbage Collections on the main thread.

**Evidence**:
```java
// ImprovedSwipeGestureRecognizer.java
public void addPoint(float x, float y, KeyboardData.Key key) {
    // ...
    _rawPath.add(new PointF(x, y)); // Allocation 1
    // ...
    PointF smoothedPoint = applySmoothing(x, y); // Allocation 2 inside method
    _smoothedPath.add(smoothedPoint);
}
```

### 2. Rendering Allocations in `Keyboard2View`

**File**: `srcs/juloo.keyboard2/Keyboard2View.java`

**Issue**:
The `onDraw` method allocates objects **every frame** while swiping.
1.  `_swipeRecognizer.getSwipePath()` returns `new ArrayList<>(_smoothedPath)`.
2.  `drawSwipeTrail` creates `new Path()` every frame.

**Impact**:
`onDraw` is critical for 60fps rendering. Allocating objects here guarantees GC churn and frame drops (jank) during the swipe animation.

**Evidence**:
```java
// Keyboard2View.java
private void drawSwipeTrail(Canvas canvas) {
    List<PointF> swipePath = _swipeRecognizer.getSwipePath(); // Allocation: new ArrayList
    // ...
    Path path = new Path(); // Allocation: new Path object every frame
    // ...
    canvas.drawPath(path, _swipeTrailPaint);
}
```

## Recommendations

1.  **Integrate `TrajectoryObjectPool`**:
    *   Update `ImprovedSwipeGestureRecognizer` to use `TrajectoryObjectPool.obtainPointF()` instead of `new PointF()`.
    *   Implement a recycling mechanism to return points to the pool when the gesture ends or resets.

2.  **Optimize `drawSwipeTrail`**:
    *   Make `Path` a member variable (`private final Path _renderPath = new Path();`) and use `.rewind()` or `.reset()` instead of allocating a new one in `onDraw`.
    *   Change `getSwipePath()` to return a read-only view or allow direct access to the internal list to avoid `new ArrayList<>(...)` copying every frame.

## Estimated Improvement
Eliminating these allocations will significantly reduce GC overhead on the main thread, resulting in smoother swipe trails and more responsive touch handling.
