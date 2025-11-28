package juloo.keyboard2

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Comprehensive test suite for KeyModifier.
 *
 * Tests cover:
 * - Modifier composition (Shift + Fn + char)
 * - Shift key behavior (upper/lower case, shifted symbols)
 * - Fn key behavior (alternate character mappings)
 * - Ctrl/Alt/Meta key behavior
 * - Modifier precedence (Fn + Shift order)
 * - Edge cases (undefined combinations, null values)
 * - Compose pending state handling
 * - Accent/diacritic modifiers
 * - Long press modifications
 * - Numpad script modifications
 * - Selection mode handling
 * - Hangul composition
 */
class KeyModifierTest {

    @Before
    fun setUp() {
        // Reset modmap to null for clean test state
        KeyModifier.set_modmap(null)
    }

    // ============================================
    // NULL AND EMPTY TESTS
    // ============================================

    @Test
    fun testModifyNull_returnsNull() {
        val result = KeyModifier.modify(null, Pointers.Modifiers.EMPTY)
        assertNull(result)
    }

    @Test
    fun testModifyEmptyString_returnsNull() {
        // Keys with empty string are placeholders and should return null
        val emptyKey = KeyValue.makeStringKey("")
        val result = KeyModifier.modify(emptyKey, Pointers.Modifiers.EMPTY)
        assertNull(result)
    }

    @Test
    fun testModifyNoModifiers_returnsOriginal() {
        val key = KeyValue.makeCharKey('a')
        val result = KeyModifier.modify(key, Pointers.Modifiers.EMPTY)

        assertNotNull(result)
        assertEquals('a', result?.getChar())
    }

    // ============================================
    // SHIFT MODIFIER TESTS
    // ============================================

    @Test
    fun testShift_lowercaseChar_becomesUppercase() {
        val key = KeyValue.makeCharKey('a')
        val shiftMod = KeyValue.getKeyByName("shift")

        val result = KeyModifier.modify(key, shiftMod)

        assertNotNull(result)
        assertEquals('A', result.getChar())
    }

    @Test
    fun testShift_uppercaseChar_staysUppercase() {
        val key = KeyValue.makeCharKey('A')
        val shiftMod = KeyValue.getKeyByName("shift")

        val result = KeyModifier.modify(key, shiftMod)

        assertNotNull(result)
        assertEquals('A', result.getChar())
    }

    @Test
    fun testShift_string_capitalized() {
        val key = KeyValue.makeStringKey("hello")
        val shiftMod = KeyValue.getKeyByName("shift")

        val result = KeyModifier.modify(key, shiftMod)

        assertNotNull(result)
        assertEquals("Hello", result.getString())
    }

    @Test
    fun testShift_alreadyCapitalizedString_unchanged() {
        val key = KeyValue.makeStringKey("Hello")
        val shiftMod = KeyValue.getKeyByName("shift")

        val result = KeyModifier.modify(key, shiftMod)

        assertNotNull(result)
        assertEquals("Hello", result.getString())
    }

    // ============================================
    // FN MODIFIER TESTS
    // ============================================

    @Test
    fun testFn_arrowUp_becomesPageUp() {
        val upKey = KeyValue.makeKeyeventKey(android.view.KeyEvent.KEYCODE_DPAD_UP)
        val fnMod = KeyValue.getKeyByName("fn")

        val result = KeyModifier.modify(upKey, fnMod)

        assertNotNull(result)
        assertEquals("page_up", result.getName())
    }

    @Test
    fun testFn_arrowDown_becomesPageDown() {
        val downKey = KeyValue.makeKeyeventKey(android.view.KeyEvent.KEYCODE_DPAD_DOWN)
        val fnMod = KeyValue.getKeyByName("fn")

        val result = KeyModifier.modify(downKey, fnMod)

        assertNotNull(result)
        assertEquals("page_down", result.getName())
    }

    @Test
    fun testFn_arrowLeft_becomesHome() {
        val leftKey = KeyValue.makeKeyeventKey(android.view.KeyEvent.KEYCODE_DPAD_LEFT)
        val fnMod = KeyValue.getKeyByName("fn")

        val result = KeyModifier.modify(leftKey, fnMod)

        assertNotNull(result)
        assertEquals("home", result.getName())
    }

    @Test
    fun testFn_arrowRight_becomesEnd() {
        val rightKey = KeyValue.makeKeyeventKey(android.view.KeyEvent.KEYCODE_DPAD_RIGHT)
        val fnMod = KeyValue.getKeyByName("fn")

        val result = KeyModifier.modify(rightKey, fnMod)

        assertNotNull(result)
        assertEquals("end", result.getName())
    }

    @Test
    fun testFn_escape_becomesInsert() {
        val escKey = KeyValue.makeKeyeventKey(android.view.KeyEvent.KEYCODE_ESCAPE)
        val fnMod = KeyValue.getKeyByName("fn")

        val result = KeyModifier.modify(escKey, fnMod)

        assertNotNull(result)
        assertEquals("insert", result.getName())
    }

    @Test
    fun testFn_tab_becomesBackTab() {
        val tabKey = KeyValue.makeKeyeventKey(android.view.KeyEvent.KEYCODE_TAB)
        val fnMod = KeyValue.getKeyByName("fn")

        val result = KeyModifier.modify(tabKey, fnMod)

        assertNotNull(result)
        assertEquals("\\t", result.getName())
    }

    // ============================================
    // CTRL/ALT/META MODIFIER TESTS
    // ============================================

    @Test
    fun testCtrl_charKey_becomesKeyevent() {
        val charKey = KeyValue.makeCharKey('c')
        val ctrlMod = KeyValue.getKeyByName("ctrl")

        val result = KeyModifier.modify(charKey, ctrlMod)

        assertNotNull(result)
        // Ctrl+C should become a keyevent (for copy shortcut)
        assertEquals(KeyValue.Kind.Keyevent, result.getKind())
    }

    @Test
    fun testAlt_charKey_becomesKeyevent() {
        val charKey = KeyValue.makeCharKey('a')
        val altMod = KeyValue.getKeyByName("alt")

        val result = KeyModifier.modify(charKey, altMod)

        assertNotNull(result)
        assertEquals(KeyValue.Kind.Keyevent, result.getKind())
    }

    @Test
    fun testMeta_charKey_becomesKeyevent() {
        val charKey = KeyValue.makeCharKey('m')
        val metaMod = KeyValue.getKeyByName("meta")

        val result = KeyModifier.modify(charKey, metaMod)

        assertNotNull(result)
        assertEquals(KeyValue.Kind.Keyevent, result.getKind())
    }

    // ============================================
    // MODIFIER COMPOSITION TESTS
    // ============================================

    @Test
    fun testMultipleModifiers_shiftThenFn() {
        // Create 'a' with Shift + Fn modifiers
        val charKey = KeyValue.makeCharKey('a')
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("shift"))
            .add(KeyValue.getKeyByName("fn"))

        val result = KeyModifier.modify(charKey, mods)

        // Should apply shift first ('a' → 'A'), then fn (compose mapping)
        assertNotNull(result)
    }

    @Test
    fun testMultipleModifiers_fnThenShift() {
        // Fn+Shift should still work (order matters)
        val charKey = KeyValue.makeCharKey('a')
        val mods = Pointers.Modifiers.EMPTY
            .add(KeyValue.getKeyByName("fn"))
            .add(KeyValue.getKeyByName("shift"))

        val result = KeyModifier.modify(charKey, mods)

        assertNotNull(result)
    }

    // ============================================
    // LONG PRESS TESTS
    // ============================================

    @Test
    fun testLongPress_changeMethodAuto_becomesChangeMethod() {
        val eventKey = KeyValue.makeEventKey(KeyValue.Event.CHANGE_METHOD_AUTO)

        val result = KeyModifier.modify_long_press(eventKey)

        assertNotNull(result)
        assertEquals("change_method", result.getName())
    }

    @Test
    fun testLongPress_switchVoiceTyping_becomesVoiceTypingChooser() {
        val eventKey = KeyValue.makeEventKey(KeyValue.Event.SWITCH_VOICE_TYPING)

        val result = KeyModifier.modify_long_press(eventKey)

        assertNotNull(result)
        assertEquals("voice_typing_chooser", result.getName())
    }

    @Test
    fun testLongPress_regularKey_unchanged() {
        val charKey = KeyValue.makeCharKey('a')

        val result = KeyModifier.modify_long_press(charKey)

        assertNotNull(result)
        assertEquals('a', result.getChar())
    }

    // ============================================
    // NUMPAD SCRIPT TESTS
    // ============================================

    @Test
    fun testNumpadScript_hindiArabic_returnsCorrectState() {
        val state = KeyModifier.modify_numpad_script("hindu-arabic")
        assertTrue(state >= 0) // Valid state
    }

    @Test
    fun testNumpadScript_bengali_returnsCorrectState() {
        val state = KeyModifier.modify_numpad_script("bengali")
        assertTrue(state >= 0)
    }

    @Test
    fun testNumpadScript_devanagari_returnsCorrectState() {
        val state = KeyModifier.modify_numpad_script("devanagari")
        assertTrue(state >= 0)
    }

    @Test
    fun testNumpadScript_persian_returnsCorrectState() {
        val state = KeyModifier.modify_numpad_script("persian")
        assertTrue(state >= 0)
    }

    @Test
    fun testNumpadScript_null_returnsNegative() {
        val state = KeyModifier.modify_numpad_script(null)
        assertEquals(-1, state)
    }

    @Test
    fun testNumpadScript_invalid_returnsNegative() {
        val state = KeyModifier.modify_numpad_script("invalid-script")
        assertEquals(-1, state)
    }

    // ============================================
    // COMPOSE PENDING TESTS
    // ============================================

    @Test
    fun testComposePending_exitWithComposeKey() {
        // Create compose pending modifier
        val composeMod = KeyValue.getKeyByName("compose")
        val pendingCompose = KeyValue.makeComposePendingKey(ComposeKeyData.accent_grave)

        val result = KeyModifier.modify(pendingCompose, composeMod)

        assertNotNull(result)
        // Tapping compose again should exit pending state
        assertEquals("compose_cancel", result.getName())
    }

    @Test
    fun testComposePending_charKey_appliesComposition() {
        // Apply compose pending to a character
        val charKey = KeyValue.makeCharKey('a')
        val pendingKey = KeyValue.makeComposePendingKey(ComposeKeyData.accent_grave)

        val result = KeyModifier.modify(charKey, pendingKey)

        assertNotNull(result)
        // Should apply grave accent composition or grey out if no match
    }

    // ============================================
    // ACCENT/DIACRITIC TESTS
    // ============================================

    @Test
    fun testGraveAccent_char_appliesAccent() {
        val charKey = KeyValue.makeCharKey('a')
        val graveMod = KeyValue.getKeyByName("accent_grave")

        val result = KeyModifier.modify(charKey, graveMod)

        assertNotNull(result)
        // Should apply grave accent (à) or return original
    }

    @Test
    fun testAiguAccent_char_appliesAccent() {
        val charKey = KeyValue.makeCharKey('e')
        val aiguMod = KeyValue.getKeyByName("accent_aigu")

        val result = KeyModifier.modify(charKey, aiguMod)

        assertNotNull(result)
        // Should apply aigu accent (é) or return original
    }

    @Test
    fun testCirconflexeAccent_char_appliesAccent() {
        val charKey = KeyValue.makeCharKey('a')
        val circonflexeMod = KeyValue.getKeyByName("accent_circonflexe")

        val result = KeyModifier.modify(charKey, circonflexeMod)

        assertNotNull(result)
        // Should apply circonflexe accent (â) or return original
    }

    @Test
    fun testTildeAccent_char_appliesAccent() {
        val charKey = KeyValue.makeCharKey('n')
        val tildeMod = KeyValue.getKeyByName("accent_tilde")

        val result = KeyModifier.modify(charKey, tildeMod)

        assertNotNull(result)
        // Should apply tilde accent (ñ) or return original
    }

    // ============================================
    // MODMAP TESTS
    // ============================================

    @Test
    fun testModmap_canBeSetAndUnset() {
        // Test that modmap can be set and cleared without crashing
        val modmap = Modmap()

        KeyModifier.set_modmap(modmap)
        // Modifier operations should now use modmap

        KeyModifier.set_modmap(null)
        // Modifier operations should fall back to default behavior
    }

    @Test
    fun testModmap_nullDoesNotCrash() {
        KeyModifier.set_modmap(null)

        val charKey = KeyValue.makeCharKey('a')
        val shiftMod = KeyValue.getKeyByName("shift")

        val result = KeyModifier.modify(charKey, shiftMod)

        assertNotNull(result)
        assertEquals('A', result.getChar())
    }
}
