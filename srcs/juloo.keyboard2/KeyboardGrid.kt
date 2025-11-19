package juloo.keyboard2

import android.graphics.PointF

/**
 * QWERTY keyboard grid matching Python KeyboardGrid exactly.
 *
 * Uses normalized [0,1] coordinates with:
 * - 3 rows (height = 1/3 each)
 * - 10 keys per row width (key_w = 0.1)
 * - Row offsets: top=0.0, mid=0.05, bot=0.15
 *
 * This grid is used for nearest key detection during swipe typing.
 * The model was trained on this specific layout, so inference must match.
 */
object KeyboardGrid {

    // Key dimensions in normalized space
    private const val KEY_WIDTH = 0.1f      // 1/10
    private const val ROW_HEIGHT = 1f / 3f  // 3 rows

    // Row definitions
    private val ROW_0 = "qwertyuiop"  // 10 keys, offset 0.0
    private val ROW_1 = "asdfghjkl"   // 9 keys, offset 0.05
    private val ROW_2 = "zxcvbnm"     // 7 keys, offset 0.15

    private val ROW_0_OFFSET = 0.0f
    private val ROW_1_OFFSET = 0.05f
    private val ROW_2_OFFSET = 0.15f

    // Pre-computed key centers for fast lookup
    private val keyPositions: Map<Char, PointF> = buildKeyPositions()

    private fun buildKeyPositions(): Map<Char, PointF> {
        val positions = mutableMapOf<Char, PointF>()

        // Row 0: qwertyuiop
        for (i in ROW_0.indices) {
            val cx = ROW_0_OFFSET + i * KEY_WIDTH + KEY_WIDTH / 2f
            val cy = 0f * ROW_HEIGHT + ROW_HEIGHT / 2f
            positions[ROW_0[i]] = PointF(cx, cy)
        }

        // Row 1: asdfghjkl
        for (i in ROW_1.indices) {
            val cx = ROW_1_OFFSET + i * KEY_WIDTH + KEY_WIDTH / 2f
            val cy = 1f * ROW_HEIGHT + ROW_HEIGHT / 2f
            positions[ROW_1[i]] = PointF(cx, cy)
        }

        // Row 2: zxcvbnm
        for (i in ROW_2.indices) {
            val cx = ROW_2_OFFSET + i * KEY_WIDTH + KEY_WIDTH / 2f
            val cy = 2f * ROW_HEIGHT + ROW_HEIGHT / 2f
            positions[ROW_2[i]] = PointF(cx, cy)
        }

        return positions
    }

    /**
     * Find nearest key to a normalized [0,1] coordinate.
     *
     * @param nx Normalized x coordinate [0,1]
     * @param ny Normalized y coordinate [0,1]
     * @return Nearest character key
     */
    fun getNearestKey(nx: Float, ny: Float): Char {
        // Clamp to valid range
        val x = nx.coerceIn(0f, 1f)
        val y = ny.coerceIn(0f, 1f)

        var nearestKey = 'a'
        var minDist = Float.MAX_VALUE

        for ((key, pos) in keyPositions) {
            val dx = x - pos.x
            val dy = y - pos.y
            val dist = dx * dx + dy * dy  // Squared distance (no sqrt needed)

            if (dist < minDist) {
                minDist = dist
                nearestKey = key
            }
        }

        return nearestKey
    }

    /**
     * Convert character to token index.
     * a-z → 4-29, others → 0 (PAD)
     */
    fun charToTokenIndex(c: Char): Int {
        return if (c in 'a'..'z') {
            (c - 'a') + 4
        } else {
            0 // PAD_IDX
        }
    }

    /**
     * Find nearest key and return token index.
     */
    fun getNearestKeyToken(nx: Float, ny: Float): Int {
        return charToTokenIndex(getNearestKey(nx, ny))
    }

    /**
     * Get key center position for a character.
     */
    fun getKeyPosition(c: Char): PointF? {
        return keyPositions[c]
    }

    /**
     * Debug: Log all key positions
     */
    fun logKeyPositions() {
        for ((key, pos) in keyPositions.entries.sortedBy { it.key }) {
            android.util.Log.d("KeyboardGrid", "$key -> (${pos.x}, ${pos.y})")
        }
    }
}
