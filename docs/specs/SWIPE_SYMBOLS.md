# Swipe Symbols Specification

**Version**: 1.0
**Status**: Documented - Small hit zones for some directions

---

## Overview

Unexpected Keyboard supports 8-directional swipe gestures to access symbols around each key. However, some directions (particularly NE and SE) have small hit zones requiring precise swipes.

### Known UX Issues

**Small Hit Zones**:
- **NE symbols** (like `-` on `g`, `=` on `h`): Must swipe precisely between N and NE
- **SE symbols** (like `]` on `k`, `}` on `j`): Must swipe precisely between S and SE
- **Other directions** (SW, S, N, etc.): Work reliably with larger tolerance

**Root Cause**: Direction-to-symbol mapping concentrates some positions into narrow angle ranges

---

## Architecture

### Components

1. **Layout XML** (`srcs/layouts/*.xml`) - Defines symbol positions using compass points
2. **Pointers.java** - Calculates swipe direction from touch coordinates
3. **KeyboardData.java** - Parses layouts and stores key values in indexed arrays

### Key Positions

Layout XML uses compass notation:
```
nw(1)  n(7)  ne(2)
w(5)   c(0)   e(6)
sw(3)  s(8)  se(4)
```

**Note**: The numeric indices (in parentheses) don't match intuitive order due to historical reasons.

---

## Direction Calculation

### File: `srcs/juloo.keyboard2/Pointers.java`

**Short Gesture Direction** (lines 241-244):
```java
double a = Math.atan2(dy, dx) + Math.PI;
// a is between 0 and 2pi, 0 is pointing to the left
// add 12 to align 0 to the top
int direction = ((int)(a * 8 / Math.PI) + 12) % 16;
```

**Process**:
1. `atan2(dy, dx)` - Angle in radians (0° = right, 90° = down in Android screen coords)
2. `+ Math.PI` - Rotate so 0 is at left (180°)
3. `* 8 / Math.PI` - Convert to [0, 16) range (16 directions)
4. `+ 12` - Rotate so direction 0 is at top (north)
5. `% 16` - Wrap around

**Result**: Direction 0-15 representing 16 sectors of a circle, clockwise from north.

---

## Direction to Position Mapping

### File: `srcs/juloo.keyboard2/Pointers.java` (line 358-360)

```java
static final int[] DIRECTION_TO_INDEX = new int[]{
  7, 2, 2, 6, 6, 4, 4, 8, 8, 3, 3, 5, 5, 1, 1, 7
};
```

**Mapping Table**:

| Direction | Angle Range | Index | Position | Symbol Example |
|-----------|-------------|-------|----------|----------------|
| 0         | 0° (N)      | 7     | n        | Numbers 1-0    |
| 1         | 22.5°       | 2     | **ne**   | **-, =, \|**   |
| 2         | 45° (NE)    | 2     | **ne**   | **-, =, \|**   |
| 3         | 67.5°       | 6     | sw       | _, +, \\       |
| 4         | 90° (E)     | 6     | sw       | _, +, \\       |
| 5         | 112.5°      | 4     | **se**   | **], }, )**    |
| 6         | 135° (SE)   | 4     | **se**   | **], }, )**    |
| 7         | 157.5°      | 8     | nw       | @, !, ~        |
| 8         | 180° (S)    | 8     | nw       | @, !, ~        |
| 9         | 202.5°      | 3     | e        | (rare)         |
| 10        | 225° (SW)   | 3     | e        | (rare)         |
| 11        | 247.5°      | 5     | s        | Numbers 1-0    |
| 12        | 270° (W)    | 5     | s        | Numbers 1-0    |
| 13        | 292.5°      | 1     | w        | (rare)         |
| 14        | 315° (NW)   | 1     | w        | (rare)         |
| 15        | 337.5°      | 7     | n        | Numbers 1-0    |

**Key Observations**:
- **NE position (index 2)**: Only directions 1-2 (22.5°-45°) = **22.5° range** ⚠️
- **SE position (index 4)**: Only directions 5-6 (112.5°-135°) = **22.5° range** ⚠️
- **N position (index 7)**: Directions 0, 15 (0°, 337.5°) = **45° range** ✓
- **S position (index 5)**: Directions 11-12 (247.5°-270°) = **45° range** ✓

This explains why NE and SE symbols are hard to hit!

---

## Nearest Direction Search

When exact direction doesn't have a symbol, `getNearestKeyAtDirection()` searches nearby:

### File: `srcs/juloo.keyboard2/Pointers.java` (line 378-401)

```java
// [i] is [0, -1, +1, ..., -3, +3], scanning 43% of the circle's area
for (int i = 0; i > -4; i = (~i>>31) - i)
{
  int d = (direction + i + 16) % 16;
  k = _handler.modifyKey(getKeyAtDirection(ptr.key, d), ptr.modifiers);
  if (k != null)
    return k;
}
```

**Search Pattern**: `[0, -1, +1, -2, +2, -3, +3]`

**Example**: If swipe calculates direction 3:
1. Check direction 3 → index 6 (sw)
2. Check direction 2 → index 2 (ne)
3. Check direction 4 → index 6 (sw)
4. Check direction 1 → index 2 (ne)
5. etc.

**Problem**: Direction 3 is closer to SW than NE in the array, so SW might be selected even when user intended NE.

---

## Why This Mapping Exists

The mapping appears designed to **balance between opposite positions**:
- Directions 1-2 (NE) vs 7-8 (opposite side)
- Directions 5-6 (SE) vs 3-4 (opposite side)

This prevents accidental triggering of opposite symbols, but creates **small hit zones** for NE and SE.

---

## Examples

### Example 1: Dash on `g`

**Layout**: `<key c="g" ne="-" sw="_"/>`

**Hit Zone for `-` (ne)**:
- Must swipe between 22.5° and 45° (very narrow!)
- Too far left (0°-22.5°) = no symbol
- Too far right (45°+) = may hit `_` (sw) via fallback search

**Hit Zone for `_` (sw)**:
- Swipe 67.5°-90° (easier, wider angle)

### Example 2: Bracket on `k`

**Layout**: `<key c="k" sw="[" se="]"/>`

**Hit Zone for `]` (se)**:
- Must swipe between 112.5° and 135° (narrow!)
- Slightly off = may find `[` (sw) via search

**Hit Zone for `[` (sw)**:
- Swipe 67.5°-90° (easier)

### Example 3: Numbers (work well)

**Layout**: `<key c="o" ne="9" sw="(" se=")"/>`

**Hit Zone for `9` (n position on other keys)**:
- Swipe 0° or 337.5°-22.5° (45° total range!)
- Much easier to hit

---

## Workarounds

### For Users

1. **Precise Swipes**: Swipe exactly between N and NE for NE symbols
2. **Slower Swipes**: Slower movements may improve accuracy
3. **Settings**: Adjust "Short Gesture Min Distance" in Settings → Swipe typing

### For Developers

**Option 1: Expand NE/SE Hit Zones**

Change DIRECTION_TO_INDEX to give NE/SE more directions:
```java
// Current: NE gets dirs 1-2, SE gets dirs 5-6 (22.5° each)
// Proposed: NE gets dirs 1-3, SE gets dirs 5-7 (45° each)
static final int[] DIRECTION_TO_INDEX = new int[]{
  7, 2, 2, 2, 6, 4, 4, 4, 8, 3, 3, 5, 5, 1, 1, 7
  // Added one more "2" at position 3, one more "4" at position 7
};
```

**Trade-off**: May make SW harder to hit

**Option 2: Adjust Search Pattern**

Modify `getNearestKeyAtDirection` to prefer certain directions:
- When searching from direction 3-4, check NE before SW
- Bias toward diagonal positions (NE, SE, SW, NW)

**Option 3: User-Configurable Sensitivity**

Add setting to control hit zone size (how many adjacent directions count)

---

## Testing

### Test Cases

1. **NE symbols** (-, =, |):
   - Swipe 30° (between N and NE) → Should work
   - Swipe 45° (exactly NE) → Should work
   - Swipe 50° → May fail (too close to SW range)

2. **SE symbols** (], }, )):
   - Swipe 120° (between E and SE) → Should work
   - Swipe 135° (exactly SE) → Should work
   - Swipe 100° → May fail (SW range)

3. **SW symbols** (_, +, \\):
   - Swipe 67.5°-90° → Should work reliably
   - Wide hit zone, easy to trigger

4. **N/S symbols** (numbers):
   - Swipe 0°±22.5° (N) → Should work
   - Swipe 270°±22.5° (S) → Should work
   - Wide hit zones

---

## Configuration

### Settings

**Short Gesture Min Distance** (Settings → Swipe typing):
- Default: 9%
- Range: 0-100%
- Higher = requires longer swipe = more accurate direction
- Lower = shorter swipe needed = may be less accurate

---

## Future Improvements

1. **Expand NE/SE zones** to 45° each (same as N/S)
2. **Visual feedback** during swipe showing which direction detected
3. **Adaptive zones** - learn from user's swipe patterns
4. **Haptic feedback** when entering symbol hit zone

---

## Version History

- **v1.32.122**: Issue identified and documented
- **Earlier**: Original implementation (upstream Unexpected Keyboard)

---

## References

- **Implementation**: `srcs/juloo.keyboard2/Pointers.java:241-244, 358-401`
- **Layout**: `srcs/layouts/latn_qwerty_us.xml`
- **Parsing**: `srcs/juloo.keyboard2/KeyboardData.java:475-494`
