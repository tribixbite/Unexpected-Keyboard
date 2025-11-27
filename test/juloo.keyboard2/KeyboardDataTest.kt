package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Test
import java.io.StringReader

/**
 * Test suite for KeyboardData to ensure keyboard layout data is properly parsed
 * and calculated, particularly keysHeight which was previously broken (v1.32.917 fix).
 */
class KeyboardDataTest {

    /**
     * Test that keysHeight is properly calculated and not 0.
     * Regression test for bug where keysHeight was hardcoded to 0f,
     * causing NaN in row_height calculations and keyboard not rendering.
     */
    @Test
    fun testKeysHeightIsCalculated() {
        // Create a simple test keyboard with 3 rows of different heights
        val testXml = """
            <?xml version="1.0" encoding="utf-8"?>
            <keyboard name="test_keyboard">
                <row height="1.0">
                    <key key0="q" width="1.0"/>
                    <key key0="w" width="1.0"/>
                </row>
                <row height="1.0">
                    <key key0="a" width="1.0"/>
                    <key key0="s" width="1.0"/>
                </row>
                <row height="1.0">
                    <key key0="z" width="1.0"/>
                    <key key0="x" width="1.0"/>
                </row>
            </keyboard>
        """.trimIndent()

        val keyboard = KeyboardData.load_string(testXml)

        // Verify keysHeight is not 0 (was the bug)
        assertTrue("keysHeight should be greater than 0", keyboard.keysHeight > 0f)

        // Verify it equals the sum of row heights
        // Each row has height=1.0, and there are 3 rows
        assertEquals("keysHeight should equal sum of row heights", 3.0f, keyboard.keysHeight, 0.01f)
    }

    /**
     * Test keysHeight calculation with rows that have shift values.
     */
    @Test
    fun testKeysHeightWithShift() {
        val testXml = """
            <?xml version="1.0" encoding="utf-8"?>
            <keyboard name="test_keyboard_shift">
                <row height="1.0" shift="0.5">
                    <key key0="q" width="1.0"/>
                </row>
                <row height="1.0" shift="0.3">
                    <key key0="a" width="1.0"/>
                </row>
            </keyboard>
        """.trimIndent()

        val keyboard = KeyboardData.load_string(testXml)

        // keysHeight should include shift values
        // Expected: (1.0 + 0.5) + (1.0 + 0.3) = 2.8
        assertEquals("keysHeight should include shift values", 2.8f, keyboard.keysHeight, 0.01f)
    }

    /**
     * Test that keysWidth is also properly calculated.
     */
    @Test
    fun testKeysWidthIsCalculated() {
        val testXml = """
            <?xml version="1.0" encoding="utf-8"?>
            <keyboard name="test_keyboard_width">
                <row height="1.0">
                    <key key0="q" width="1.0"/>
                    <key key0="w" width="1.5"/>
                    <key key0="e" width="2.0"/>
                </row>
            </keyboard>
        """.trimIndent()

        val keyboard = KeyboardData.load_string(testXml)

        // keysWidth should be max row width
        // Row width = 1.0 + 1.5 + 2.0 = 4.5
        assertTrue("keysWidth should be greater than 0", keyboard.keysWidth > 0f)
        assertEquals("keysWidth should equal max row width", 4.5f, keyboard.keysWidth, 0.01f)
    }

    /**
     * Test that empty keyboard doesn't crash and has reasonable defaults.
     */
    @Test
    fun testEmptyKeyboardHandling() {
        val testXml = """
            <?xml version="1.0" encoding="utf-8"?>
            <keyboard name="test_empty">
            </keyboard>
        """.trimIndent()

        val keyboard = KeyboardData.load_string(testXml)

        // Should have 0 rows and sensible defaults
        assertEquals("Empty keyboard should have 0 rows", 0, keyboard.rows.size)
        // keysHeight should be 0 for empty keyboard (valid case)
        assertEquals("Empty keyboard keysHeight should be 0", 0f, keyboard.keysHeight, 0.01f)
    }
}
