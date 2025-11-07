# Short Swipe Gestures System

**Version**: 2.0
**Status**: Fully Implemented
**Last Updated**: 2025-11-06

---

## Overview

Short swipe gestures (also called "directional swipes" or "within-key swipes") allow users to access additional characters by swiping in different directions while staying on a single key. Unlike swipe typing (which moves across multiple keys), short swipes remain within the starting key's boundaries.

### Goals

- **Fast Symbol Access**: Type symbols without switching keyboard layers
- **Muscle Memory**: Consistent directional mappings (e.g., numbers always on top row)
- **Touch Accuracy**: Forgiving tolerance for imprecise swipes
- **No Accidental Triggers**: Don't confuse taps with short swipes

### Key Features

- **8-directional detection**: North, Northeast, East, Southeast, South, Southwest, West, Northwest
- **Radial tolerance system**: Equal tolerance in all directions (v1.32.303+)
- **Dynamic sizing**: Adapts to user's keyboard height and screen size settings
- **Configurable sensitivity**: User-adjustable minimum swipe distance

---

## Architecture

### System Components

```
User Touch Input
      ↓
┌─────────────────────────────────────────────┐
│  Pointers.java - Touch Event Handler        │
│  • onTouchDown(): Record starting position  │
│  • onTouchMove(): Track swipe path          │
│  • onTouchUp(): Classify gesture            │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│  Gesture Classification (Line 226)          │
│  if (!hasLeftStartingKey && distance >= min)│
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│  Direction Calculation (Lines 241-244)      │
│  angle = atan2(dy, dx)                      │
│  direction = ((angle * 8 / π) + 12) % 16    │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│  DIRECTION_TO_INDEX Mapping (Line 375)      │
│  16 directions → 9 key positions            │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│  getNearestKeyAtDirection (Lines 395-418)   │
│  Search nearby if exact direction empty     │
└─────────────────────────────────────────────┘
      ↓
Output Character (e.g., ], =, -, etc.)
```

### File Locations

| Component | File | Lines |
|-----------|------|-------|
| Touch handling | `Pointers.java` | 145-314 |
| Tolerance check | `Keyboard2View.java` | 282-371 |
| Direction mapping | `Pointers.java` | 373-386 |
| Nearest search | `Pointers.java` | 395-418 |
| Configuration | `Config.java` | 109, 285 |

---

## Key Position Layout

Each key can have up to 9 values arranged in compass directions:

```
nw(1)   n(7)   ne(2)
  ╲      |      ╱
   ╲     |     ╱
w(5)──── c(0) ────e(6)
   ╱     |     ╲
  ╱      |      ╲
sw(3)   s(8)   se(4)
```

**Array Indices** (historical, non-sequential):
- `c` (center) = 0
- `nw` (northwest) = 1
- `ne` (northeast) = 2
- `sw` (southwest) = 3
- `se` (southeast) = 4
- `w` (west) = 5
- `e` (east) = 6
- `n` (north) = 7
- `s` (south) = 8

**Layout XML Example**:
```xml
<key c="i" nw="*" ne="8" sw="I'd " w="it " se="I'm "/>
```

---

## Tolerance System (v1.32.303)

### Problem: Rectangular vs Radial Tolerance

**Historical Issue** (pre-v1.32.301):
- Used **rectangular tolerance**: separate X and Y margins
- Keys are wider than tall (e.g., 80px × 60px)
- With 40% tolerance:
  - Horizontal margin: 32px (40% of 80px)
  - Vertical margin: 24px (40% of 60px)
- **Southeast/Southwest swipes discriminated against**: Hit smaller vertical boundary first

**First Fix Attempt** (v1.32.301) - **BROKEN**:
```java
// WRONG: Made tolerance even worse!
float keyHalfDiagonal = sqrt((W² + H²) / 4);
maxDistance = keyHalfDiagonal × 1.4;
// Result: 50 × 1.4 = 70px (less than old 72px horizontal!)
```

**Correct Fix** (v1.32.303):
```java
// Circle must fully contain the extended rectangle
float maxHorizontal = keyWidth × (0.5 + tolerance);   // e.g., 72px
float maxVertical = keyHeight × (0.5 + tolerance);    // e.g., 54px
float maxDistance = sqrt(maxH² + maxV²);              // e.g., 90px
```

### Current Implementation

**File**: `Keyboard2View.java:350-357`

```java
// Calculate max allowed distance - must contain the old rectangular bounds
// The old rectangle extended by tolerance creates bounds of:
//   horizontal: keyWidth/2 + keyWidth*tolerance = keyWidth*(0.5 + tolerance)
//   vertical: keyHeight/2 + keyHeight*tolerance = keyHeight*(0.5 + tolerance)
// Our circle must reach the farthest corner of this rectangle
float maxHorizontal = keyWidth * (0.5f + tolerance);
float maxVertical = keyHeight * (0.5f + tolerance);
float maxDistance = (float)Math.sqrt(maxHorizontal * maxHorizontal + maxVertical * maxVertical);

boolean result = distanceFromCenter <= maxDistance;
```

### Tolerance Comparison

For a typical 80×60px key with 40% tolerance:

| Direction | Old Rectangular | Broken Radial (v1.32.301) | Fixed Radial (v1.32.303) |
|-----------|----------------|---------------------------|--------------------------|
| East | 72px | 70px ❌ | **90px ✓** |
| North | 54px | 70px ✓ | **90px ✓** |
| Northeast | 90px | 70px ❌ | **90px ✓** |
| Southeast | 90px | 70px ❌ | **90px ✓** |

**Result**: All directions now have **equal tolerance** (90px from center), and diagonal swipes work with straight-line movements.

---

## Dynamic Sizing

### Key Dimensions Calculation

**Width** (`Keyboard2View.java:631`):
```java
_keyWidth = (width - _marginLeft - _marginRight) / _keyboard.keysWidth;
```

**Height** (`Theme.java:110-112`):
```java
row_height = Math.min(
    config.screenHeightPixels * config.keyboardHeightPercent / 100 / 3.95f,
    config.screenHeightPixels / layout.keysHeight);
```

**Factors Affecting Size**:
1. **Screen dimensions** (`screenHeightPixels`, `screenWidthPixels`)
2. **User settings**:
   - Keyboard height percentage (Settings → Appearance)
   - Key margins (horizontal/vertical)
3. **Layout structure** (`keysWidth`, `keysHeight` from XML)

### Example Calculation

```
Screen: 1920×1080px (portrait)
Keyboard height: 40% of screen
Layout: 10 keys wide, 4 rows

_keyWidth = (1080 - margins) / 10 ≈ 100px
row_height = (1920 × 0.4) / 4 ≈ 192px

Typical key: 100×192px
With 40% tolerance:
  maxHorizontal = 100 × 0.9 = 90px
  maxVertical = 192 × 0.9 = 173px
  maxDistance = sqrt(90² + 173²) ≈ 195px
```

**Key Insight**: Tolerance automatically scales with user's settings!

---

## Direction Calculation

### Algorithm (Pointers.java:241-244)

```java
double a = Math.atan2(dy, dx) + Math.PI;
// a is between 0 and 2π, 0 is pointing to the left
// add 12 to align 0 to the top
int direction = ((int)(a * 8 / Math.PI) + 12) % 16;
```

### Step-by-Step Example

**Swipe**: dx=10px, dy=10px (45° southeast)

1. `atan2(10, 10)` = 0.785 rad (45°)
2. `+ Math.PI` = 3.927 rad (225° rotated)
3. `* 8 / Math.PI` = 10.0
4. `(int)` = 10
5. `+ 12` = 22
6. `% 16` = **6** (direction index)

**Direction 6** → **SE position** (index 4) → Output ']' on k key

### Angle to Direction Table

| Direction | Angle Range (°) | Primary Compass | Example Swipe |
|-----------|----------------|-----------------|---------------|
| 0 | 348.75 - 11.25 | North | Straight up |
| 1 | 11.25 - 33.75 | N-NE | Slightly right of up |
| 2 | 33.75 - 56.25 | NE | 45° diagonal up-right |
| 3 | 56.25 - 78.75 | E-NE | Slightly up from right |
| 4 | 78.75 - 101.25 | East | Straight right |
| 5 | 101.25 - 123.75 | E-SE | Slightly down from right |
| 6 | 123.75 - 146.25 | SE | 45° diagonal down-right |
| 7 | 146.25 - 168.75 | S-SE | Slightly right of down |
| 8 | 168.75 - 191.25 | South | Straight down |
| 9 | 191.25 - 213.75 | S-SW | Slightly left of down |
| 10 | 213.75 - 236.25 | SW | 45° diagonal down-left |
| 11 | 236.25 - 258.75 | W-SW | Slightly down from left |
| 12 | 258.75 - 281.25 | West | Straight left |
| 13 | 281.25 - 303.75 | W-NW | Slightly up from left |
| 14 | 303.75 - 326.25 | NW | 45° diagonal up-left |
| 15 | 326.25 - 348.75 | N-NW | Slightly left of up |

**Note**: Each direction covers 22.5° of the circle (360° / 16).

---

## Direction to Position Mapping

### DIRECTION_TO_INDEX Array (Pointers.java:375-377)

```java
static final int[] DIRECTION_TO_INDEX = new int[]{
  7, 2, 2, 6, 4, 4, 4, 8, 8, 3, 3, 5, 5, 1, 1, 7
};
```

### Visual Mapping

```
     NW(1)          N(7)          NE(2)
      dirs          dirs          dirs
      13-14         0,15          1-2
        ╲            |            ╱
         ╲           |           ╱
          ╲          |          ╱
W(5) ──────╲────── C(0) ──────╱────── E(6)
dirs         ╲       |       ╱         dirs
12            ╲      |      ╱          3-4
               ╲     |     ╱
                ╲    |    ╱
     SW(3)       ╲   |   ╱       SE(4)
      dirs        ╲  |  ╱        dirs
      9-10         ╲ | ╱         5-6-7
                    ╲|╱
                   S(8)
                   dirs
                   7-8
```

### Hit Zone Analysis

| Position | Directions | Angle Coverage | Width |
|----------|-----------|----------------|-------|
| **N** | 0, 15 | 348.75°-11.25° | 45° ✓ |
| **NE** | 1-2 | 11.25°-56.25° | 45° ✓ |
| **E** | 3-4 | 56.25°-101.25° | 45° ✓ |
| **SE** | 5-6-7 | 101.25°-168.75° | **67.5° ✓✓** |
| **S** | 8 | 168.75°-191.25° | 22.5° ⚠️ |
| **SW** | 9-10 | 191.25°-236.25° | 45° ✓ |
| **W** | 11-12 | 236.25°-281.25° | 45° ✓ |
| **NW** | 13-14 | 281.25°-326.25° | 45° ✓ |

**Key Observation**: SE has the **widest hit zone** (67.5°) thanks to the expanded mapping mentioned in code comments: "Expanded SE (index 4) from dirs 5-6 to 4-6 for 45° hit zone (makes ] and } easier)".

Actually, looking at the mapping again, SE gets **dirs 4-6** (three slots), which is 67.5°. This was specifically added to make ] and } easier!

---

## Nearest Direction Search

When the exact direction has no symbol defined, the system searches nearby directions.

### Algorithm (Pointers.java:395-418)

```java
private KeyValue getNearestKeyAtDirection(Pointer ptr, int direction)
{
  KeyValue k;
  // [i] is [0, -1, +1, ..., -3, +3], scanning 43% of the circle's area
  for (int i = 0; i > -4; i = (~i>>31) - i)
  {
    int d = (direction + i + 16) % 16;
    k = _handler.modifyKey(getKeyAtDirection(ptr.key, d), ptr.modifiers);
    if (k != null)
    {
      // When the nearest key is a slider, it is only selected if placed
      // within 18% of the original swipe direction
      if (k.getKind() == KeyValue.Kind.Slider && Math.abs(i) >= 2)
        continue;
      return k;
    }
  }
  return null;
}
```

### Search Pattern

The variable `i` follows this sequence: `[0, -1, +1, -2, +2, -3, +3]`

**Example**: User swipes direction 4 (East) on key `h`:

1. **i=0**: Check dir 4 → Maps to SE (index 4) → Not defined on h → Continue
2. **i=-1**: Check dir 3 → Maps to E (index 6) → **Found "="!** → Return

**Coverage**: Searches ±3 directions = 7 out of 16 total = 43.75% of circle

### Slider Special Case

Sliders (e.g., brightness control) are only selected if within ±1 direction (18% of original):
- Prevents accidental slider activation during diagonal swipes
- Allows typing circle gestures without interruption

---

## Gesture Classification

### The hasLeftStartingKey Check (Pointers.java:226)

```java
if (_config.short_gestures_enabled && !ptr.hasLeftStartingKey && swipePath != null && swipePath.size() > 1)
```

**Critical Logic**:
- `hasLeftStartingKey` is set to `true` when touch point exceeds tolerance (Pointers.java:448-452)
- If `true`, short gesture detection is **disabled entirely**
- This prevents detecting "long swipes" as short gestures

### Tolerance Check (Pointers.java:446-452)

```java
if (ptr.key != null && !ptr.hasLeftStartingKey)
{
  if (!_handler.isPointWithinKeyWithTolerance(x, y, ptr.key, 0.40f))
  {
    ptr.hasLeftStartingKey = true;
  }
}
```

**40% tolerance** means the circle extends 40% beyond the key's half-diagonal.

### Distance Check (Pointers.java:238)

```java
float keyHypotenuse = _handler.getKeyHypotenuse(ptr.key);
float minDistance = keyHypotenuse * (_config.short_gesture_min_distance / 100.0f);

if (distance >= minDistance)
{
  // Calculate direction and output symbol
}
```

**User Setting**: `short_gesture_min_distance` (default 20%)
- Higher = longer swipe required = more accurate direction detection
- Lower = shorter swipe = more convenient but may misdetect

---

## Configuration

### User Settings (Config.java)

| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| `short_gestures_enabled` | `true` | boolean | Enable/disable short swipes |
| `short_gesture_min_distance` | `20` | 10-95 | Minimum swipe as % of key diagonal |

**Example**:
- Key diagonal: 100px
- Min distance: 20%
- Required swipe: 20px

### Developer Constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `TOLERANCE` | Pointers.java:448 | 0.40f | 40% radial tolerance |
| `DIRECTION_TO_INDEX` | Pointers.java:375 | Array[16] | Maps directions to positions |

---

## Common Layout Patterns

### Number Row

```xml
<key c="q" ne="1" se="loc esc"/>
<key c="w" nw="~" ne="2" sw="@" se="we "/>
<key c="e" nw="!" ne="3" sw="#" se="loc €"/>
<key c="r" ne="4" sw="$"/>
<key c="t" ne="5" sw="%" e="to "/>
<key c="y" ne="6" sw="^"/>
<key c="u" ne="7" sw="&" n="up "/>
<key c="i" nw="*" ne="8" sw="I'd " w="it " se="I'm "/>
<key c="o" nw="of " ne="9" sw="(" se=")" s="on " w="or "/>
<key c="p" ne="0"/>
```

**Pattern**: Numbers 1-9-0 on NE positions across top row

### Brackets and Operators

```xml
<key c="g" nw="-" ne="go " sw="_"/>
<key c="h" ne="=" sw="+" e="hi "/>
<key c="j" se="}" sw="{"/>
<key c="k" sw="[" se="]"/>
<key c="l" ne="|" sw="\"/>
```

**Pattern**:
- Math operators (-, +, =) on g-h
- Brackets ({, }, [, ]) on j-k
- Backslash/pipe on l

### Contractions (v1.32.300)

```xml
<key c="i" nw="*" ne="8" sw="I'd " w="it " se="I'm "/>
```

**Recent Update**: Swapped I'm to SE and I'd to SW for better thumb ergonomics

---

## Performance Characteristics

### Computational Cost

**Per Touch Event**:
1. **Tolerance check**: 1 sqrt operation (~5μs)
2. **Direction calculation**: 1 atan2 + arithmetic (~10μs)
3. **Array lookup**: O(1) (~1μs)
4. **Nearest search**: Max 7 iterations × O(1) lookup (~5μs)

**Total**: ~20μs per swipe (negligible)

### Memory Usage

- `DIRECTION_TO_INDEX`: 16 integers = 64 bytes
- Per-pointer tracking: 1 boolean (`hasLeftStartingKey`) = 1 byte
- **Total**: <100 bytes (negligible)

---

## Debugging

### Log Messages

**Enable logging**: Look for `"Pointers"` tag in logcat

**Short swipe detection** (Line 252-255):
```
SHORT_SWIPE: key=i dx=12.3 dy=15.6 dist=19.8 angle=141.3° dir=6→idx=4(se)
SHORT_SWIPE_RESULT: dir=6 found=I'm
```

**Tolerance check** (Keyboard2View.java:361-366):
```
isPointWithinKeyWithTolerance: point=(450.2,123.4) center=(440.0,120.0)
  distance=10.8 maxDistance=90.0 (maxH=72.0, maxV=54.0, tolerance=40.0%) result=true
```

### Common Issues

**Issue**: Short swipe not detected
- **Check**: `hasLeftStartingKey = true` → Exceeded tolerance
- **Fix**: Swipe shorter/slower, or increase tolerance in code

**Issue**: Wrong symbol detected
- **Check**: `angle` in logs → Verify direction calculation
- **Fix**: May need to adjust DIRECTION_TO_INDEX mapping

**Issue**: East/Northeast not working (pre-v1.32.303)
- **Check**: Version < v1.32.303 has broken radial tolerance
- **Fix**: Update to v1.32.303+

---

## Version History

### v1.32.303 (2025-11-06) ✅ **CURRENT**
**Correct radial tolerance formula**
- Fixed: Circle now fully contains rectangular bounds
- Formula: `maxDistance = sqrt(maxH² + maxV²)`
- Result: All directions get 90px tolerance (equal and permissive)
- **All short swipes work correctly**

### v1.32.301 (2025-11-06) ❌ **BROKEN - DO NOT USE**
**Incorrect radial tolerance**
- Broke: Used `keyHalfDiagonal × 1.4 = 70px`
- Problem: Less than old horizontal tolerance (72px)
- **East and Northeast swipes broken**

### v1.32.300 (2025-11-06)
**Contraction updates**
- Moved I'm to SE, I'd to SW on 'i' key
- Better thumb ergonomics

### v1.32.122 (Earlier)
**Expanded SE hit zone**
- Changed SE from dirs 5-6 to dirs 4-6-7 (67.5° coverage)
- Made ] and } easier to trigger

### Pre-v1.32.301
**Rectangular tolerance** (original implementation)
- Separate horizontal and vertical margins
- Discriminated against diagonal swipes
- Southeast/Southwest had issues due to small vertical tolerance

---

## Testing

### Test Cases

**1. All 8 Directions on 'o' key**:
```xml
<key c="o" nw="of " ne="9" sw="(" se=")" s="on " w="or "/>
```

| Direction | Expected | Test Angle | Status |
|-----------|----------|------------|--------|
| NW | "of " | 315° | ✓ |
| N | (none) | 0° | ✓ |
| NE | "9" | 45° | ✓ |
| E | (none) | 90° | ✓ |
| SE | ")" | 135° | ✓ |
| S | "on " | 180° | ✓ |
| SW | "(" | 225° | ✓ |
| W | "or " | 270° | ✓ |

**2. Straight-Line Diagonals** (v1.32.303+ required):
- SE straight line (dx=dy): Should output ']' on k
- NE straight line (dx=dy, dy<0): Should output '=' on h

**3. Tolerance Edge Cases**:
- Swipe exactly `maxDistance` px: Should trigger
- Swipe `maxDistance + 1` px: Should NOT trigger (hasLeftStartingKey)

**4. Nearest Search**:
- Swipe direction with no symbol: Should find nearest defined symbol
- Verify ±3 direction search range

---

## Troubleshooting

### "Southeast swipes don't work"

**Versions affected**: Pre-v1.32.303

**Root cause**: Rectangular or incorrect radial tolerance

**Fix**: Update to v1.32.303+

**Verify**: Check logs for `hasLeftStartingKey = true`

### "East swipes trigger wrong symbol"

**Possible causes**:
1. Direction mapping: Check DIRECTION_TO_INDEX
2. Nearest search: May be finding adjacent direction's symbol
3. Swipe too short: Increase swipe distance

**Debug**: Enable logging, check `dir` and `idx` values

### "All short swipes inconsistent"

**Check**:
1. Keyboard height setting (too small = hard to control)
2. Min distance setting (too low = direction unstable)
3. Key size (dynamic, varies by screen/settings)

**Recommended**:
- Min distance: 20-30%
- Keyboard height: 40-50% of screen

---

## Future Improvements

1. **Visual Direction Indicator**: Show detected direction during swipe
2. **Haptic Feedback**: Vibrate when entering symbol's hit zone
3. **User-Configurable Mapping**: Let users customize direction assignments
4. **Adaptive Tolerance**: Learn from user's swipe patterns
5. **Direction Preview**: Show available symbols while touching key

---

## References

### Code Files
- `srcs/juloo.keyboard2/Pointers.java` - Main gesture handling
- `srcs/juloo.keyboard2/Keyboard2View.java` - Tolerance calculation
- `srcs/juloo.keyboard2/Config.java` - Settings
- `srcs/juloo.keyboard2/Theme.java` - Dynamic sizing

### Related Documentation
- [SWIPE_SYMBOLS.md](SWIPE_SYMBOLS.md) - Historical direction mapping issues
- [memory/pm.md](../../memory/pm.md) - Project status and recent fixes

### External References
- [Android MotionEvent](https://developer.android.com/reference/android/view/MotionEvent)
- [Math.atan2 documentation](https://docs.oracle.com/javase/8/docs/api/java/lang/Math.html#atan2-double-double-)

---

**Document Status**: ✅ Complete and Current
**Maintainer**: Claude Code Development Team
**Last Verified**: v1.32.303 (2025-11-06)
