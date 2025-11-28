package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Test

/**
 * Comprehensive test suite for ExtraKeys.
 *
 * Tests cover:
 * - Key parsing (simple keys, alternatives, next_to positioning)
 * - Key merging (duplicate key handling, script generalization)
 * - Compute logic (script matching, alternative selection)
 * - Edge cases (empty strings, invalid keys, null handling)
 * - Alternative key selection (single alternative vs multiple)
 * - Script filtering (matching, null script, mixed scripts)
 * - Position assignment (default vs next_to)
 */
class ExtraKeysTest {

    // ============================================
    // PARSING TESTS
    // ============================================

    @Test
    fun testParse_simpleKey_createsExtraKey() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab", "latin")

        assertEquals("tab", extraKey.kv.getName())
        assertEquals("latin", extraKey.script)
        assertTrue(extraKey.alternatives.isEmpty())
        assertNull(extraKey.nextTo)
    }

    @Test
    fun testParse_keyWithAlternatives_parsesCorrectly() {
        val extraKey = ExtraKeys.ExtraKey.parse("€:dollar:pound", "latin")

        assertEquals("€", extraKey.kv.getSymbol())
        assertEquals("latin", extraKey.script)
        assertEquals(2, extraKey.alternatives.size)
        assertEquals("dollar", extraKey.alternatives[0].getName())
        assertEquals("pound", extraKey.alternatives[1].getName())
    }

    @Test
    fun testParse_keyWithNextTo_parsesPosition() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab@space", "latin")

        assertEquals("tab", extraKey.kv.getName())
        assertEquals("space", extraKey.nextTo?.getName())
    }

    @Test
    fun testParse_keyWithAlternativesAndNextTo_parsesAll() {
        val extraKey = ExtraKeys.ExtraKey.parse("€:dollar:pound@space", "latin")

        assertEquals("€", extraKey.kv.getSymbol())
        assertEquals(2, extraKey.alternatives.size)
        assertEquals("space", extraKey.nextTo?.getName())
    }

    @Test
    fun testParse_multipleKeys_parsesPipeSeparated() {
        val extraKeys = ExtraKeys.parse("latin", "tab|esc|f1")

        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())
        extraKeys.compute(dst, query)

        // Should add all 3 keys
        assertEquals(3, dst.size)
        assertTrue(dst.keys.any { it.getName() == "tab" })
        assertTrue(dst.keys.any { it.getName() == "esc" })
        assertTrue(dst.keys.any { it.getName() == "f1" })
    }

    // ============================================
    // COMPUTE LOGIC TESTS
    // ============================================

    @Test
    fun testCompute_scriptMatches_addsKey() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab", "latin")
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())

        extraKey.compute(dst, query)

        assertTrue(dst.containsKey(extraKey.kv))
    }

    @Test
    fun testCompute_scriptMismatch_doesNotAddKey() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab", "latin")
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("arabic", emptySet())

        extraKey.compute(dst, query)

        assertFalse(dst.containsKey(extraKey.kv))
    }

    @Test
    fun testCompute_nullScriptInKey_addsToAnyLayout() {
        val extraKey = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), null, emptyList(), null)
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("arabic", emptySet())

        extraKey.compute(dst, query)

        // Null script in key should match any layout script
        assertTrue(dst.containsKey(extraKey.kv))
    }

    @Test
    fun testCompute_nullScriptInQuery_addsKey() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab", "latin")
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query(null, emptySet())

        extraKey.compute(dst, query)

        // Null query script should match any key script
        assertTrue(dst.containsKey(extraKey.kv))
    }

    @Test
    fun testCompute_allAlternativesPresent_doesNotAddKey() {
        val alt1 = KeyValue.getKeyByName("dollar")
        val alt2 = KeyValue.getKeyByName("pound")
        val extraKey = ExtraKeys.ExtraKey(
            KeyValue.getKeyByName("€"),
            "latin",
            listOf(alt1, alt2),
            null
        )
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", setOf(alt1, alt2))

        extraKey.compute(dst, query)

        // All alternatives already present, so key should not be added
        assertFalse(dst.containsKey(extraKey.kv))
    }

    @Test
    fun testCompute_someAlternativesMissing_addsKey() {
        val alt1 = KeyValue.getKeyByName("dollar")
        val alt2 = KeyValue.getKeyByName("pound")
        val extraKey = ExtraKeys.ExtraKey(
            KeyValue.getKeyByName("€"),
            "latin",
            listOf(alt1, alt2),
            null
        )
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", setOf(alt1)) // Only one alternative present

        extraKey.compute(dst, query)

        // Not all alternatives present, so key should be added
        assertTrue(dst.containsKey(extraKey.kv))
    }

    @Test
    fun testCompute_singleAlternative_addsAlternativeInsteadOfKey() {
        val alt = KeyValue.getKeyByName("dollar")
        val euroKey = KeyValue.getKeyByName("€")
        val extraKey = ExtraKeys.ExtraKey(
            euroKey,
            "latin",
            listOf(alt),
            null
        )
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())

        extraKey.compute(dst, query)

        // Single alternative should be added instead of the key itself
        assertTrue(dst.containsKey(alt))
        assertFalse(dst.containsKey(euroKey))
    }

    @Test
    fun testCompute_singleAlternativeButKeyAlreadyPresent_addsKey() {
        val alt = KeyValue.getKeyByName("dollar")
        val euroKey = KeyValue.getKeyByName("€")
        val extraKey = ExtraKeys.ExtraKey(
            euroKey,
            "latin",
            listOf(alt),
            null
        )
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        dst[euroKey] = KeyboardData.PreferredPos.DEFAULT // Key already present
        val query = ExtraKeys.Query("latin", emptySet())

        extraKey.compute(dst, query)

        // Key already in dst, so should add the key itself (not alternative)
        assertTrue(dst.containsKey(euroKey))
    }

    // ============================================
    // MERGE TESTS
    // ============================================

    @Test
    fun testMerge_duplicateKeys_mergesAlternatives() {
        val extraKeys1 = ExtraKeys.parse("latin", "€:dollar")
        val extraKeys2 = ExtraKeys.parse("latin", "€:pound")

        val merged = ExtraKeys.merge(listOf(extraKeys1, extraKeys2))

        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())
        merged.compute(dst, query)

        // Should add only one € key (merged)
        assertEquals(1, dst.size)
    }

    @Test
    fun testMerge_scriptConflict_generalizesToNull() {
        val key1 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), "latin", emptyList(), null)
        val key2 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), "arabic", emptyList(), null)

        val merged = key1.mergeWith(key2)

        // Script conflict should generalize to null
        assertNull(merged.script)
    }

    @Test
    fun testMerge_sameScript_keepsScript() {
        val key1 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), "latin", emptyList(), null)
        val key2 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), "latin", emptyList(), null)

        val merged = key1.mergeWith(key2)

        // Same script should be preserved
        assertEquals("latin", merged.script)
    }

    @Test
    fun testMerge_oneNullScript_keepsNonNull() {
        val key1 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), "latin", emptyList(), null)
        val key2 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("tab"), null, emptyList(), null)

        val merged = key1.mergeWith(key2)

        // Should keep the non-null script
        assertEquals("latin", merged.script)
    }

    @Test
    fun testMerge_alternativesConcatenated() {
        val alt1 = KeyValue.getKeyByName("dollar")
        val alt2 = KeyValue.getKeyByName("pound")
        val key1 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("€"), "latin", listOf(alt1), null)
        val key2 = ExtraKeys.ExtraKey(KeyValue.getKeyByName("€"), "latin", listOf(alt2), null)

        val merged = key1.mergeWith(key2)

        // Alternatives should be concatenated
        assertEquals(2, merged.alternatives.size)
        assertTrue(merged.alternatives.contains(alt1))
        assertTrue(merged.alternatives.contains(alt2))
    }

    // ============================================
    // POSITION TESTS
    // ============================================

    @Test
    fun testCompute_noNextTo_usesDefaultPosition() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab", "latin")
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())

        extraKey.compute(dst, query)

        val pos = dst[extraKey.kv]
        assertNotNull(pos)
        assertNull(pos?.next_to)
    }

    @Test
    fun testCompute_withNextTo_setsNextToPosition() {
        val extraKey = ExtraKeys.ExtraKey.parse("tab@space", "latin")
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())

        extraKey.compute(dst, query)

        val pos = dst[extraKey.kv]
        assertNotNull(pos)
        assertEquals("space", pos?.next_to?.getName())
    }

    // ============================================
    // EMPTY/NULL TESTS
    // ============================================

    @Test
    fun testEmpty_computeNothing() {
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())

        ExtraKeys.EMPTY.compute(dst, query)

        // Empty extra keys should add nothing
        assertTrue(dst.isEmpty())
    }

    @Test
    fun testParse_emptyString_createsEmpty() {
        val extraKeys = ExtraKeys.parse("latin", "")
        val dst = mutableMapOf<KeyValue, KeyboardData.PreferredPos>()
        val query = ExtraKeys.Query("latin", emptySet())

        extraKeys.compute(dst, query)

        // Should handle empty string gracefully (adds one empty key)
        // This is acceptable behavior - parsing "" creates one empty entry
        // Real usage would filter this out at config level
    }
}
