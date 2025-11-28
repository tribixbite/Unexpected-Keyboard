# SWIPE_LAG_DEBUG Instructions

To diagnose the swipe lag, run the following command in your terminal or ADB shell while reproducing the issue (switching apps and swiping):

```bash
adb logcat -s SWIPE_LAG_DEBUG
```

**What to look for:**

1.  **❌ Touch before first key:**
    *   If you see this, it means the calculated left margin (`currentMarginLeft`) is pushing the keys too far to the right, or your touch `tx` is too far left.
    *   Pay attention to the `margin=` value. If it's unexpectedly large (e.g. from a previous landscape layout), that's the bug.

2.  **📍 'A' KEY:**
    *   This logs the calculated bounds of the 'A' key (`xLeft` - `xRight`) vs your touch `tx`.
    *   Check `kw` (key width). If `kw` is wrong (e.g. too wide), the keys will drift rightwards, causing touches to fall into "gaps" or previous keys.

3.  **❌ Touch after last key:**
    *   If your touch is valid but the logic thinks it's past the last key, the total keyboard width calculation is too small.

4.  **No logs at all:**
    *   If you swipe but see NO logs, it means `getKeyAtPosition` isn't even being called. This would imply `onTouch` isn't firing, or `getRowAtPosition` returned null (which logs an error).

Please share the output of this logcat command.