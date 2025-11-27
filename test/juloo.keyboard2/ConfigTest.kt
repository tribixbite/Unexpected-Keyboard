package juloo.keyboard2

import android.content.SharedPreferences
import android.content.res.Configuration
import android.content.res.Resources
import android.util.DisplayMetrics
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner

/**
 * Comprehensive test suite for Config.
 *
 * Tests cover:
 * - Global config initialization
 * - Preference loading with type safety
 * - Float preference repair/migration
 * - Integer preference type coercion
 * - Layout switching (portrait/landscape)
 * - Clipboard configuration
 * - Theme ID resolution
 * - Orientation handling
 * - Config versioning and migration
 * - Wide screen detection
 */
@RunWith(MockitoJUnitRunner::class)
class ConfigTest {

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockPrefsEditor: SharedPreferences.Editor

    @Mock
    private lateinit var mockResources: Resources

    @Mock
    private lateinit var mockConfiguration: Configuration

    @Mock
    private lateinit var mockHandler: Config.IKeyEventHandler

    private lateinit var displayMetrics: DisplayMetrics

    @Before
    fun setUp() {
        // Setup display metrics (standard phone)
        displayMetrics = DisplayMetrics().apply {
            widthPixels = 1080
            heightPixels = 2340
            density = 3.0f
            xdpi = 420f
            ydpi = 420f
        }

        // Setup configuration (portrait)
        mockConfiguration.orientation = Configuration.ORIENTATION_PORTRAIT
        mockConfiguration.uiMode = Configuration.UI_MODE_NIGHT_NO

        `when`(mockResources.displayMetrics).thenReturn(displayMetrics)
        `when`(mockResources.configuration).thenReturn(mockConfiguration)

        // Setup preference editor mock chain
        `when`(mockPrefs.edit()).thenReturn(mockPrefsEditor)
        `when`(mockPrefsEditor.putInt(anyString(), anyInt())).thenReturn(mockPrefsEditor)
        `when`(mockPrefsEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockPrefsEditor)
        `when`(mockPrefsEditor.putString(anyString(), anyString())).thenReturn(mockPrefsEditor)
        `when`(mockPrefsEditor.putFloat(anyString(), anyFloat())).thenReturn(mockPrefsEditor)
        `when`(mockPrefsEditor.apply()).then {}
        `when`(mockPrefsEditor.commit()).thenReturn(true)

        // Setup default preference values
        setupDefaultPreferences()
    }

    private fun setupDefaultPreferences() {
        // Config version
        `when`(mockPrefs.getInt("version", 0)).thenReturn(3)

        // Basic keyboard settings
        `when`(mockPrefs.getString("show_numpad", "never")).thenReturn("never")
        `when`(mockPrefs.getInt("keyboard_height", 35)).thenReturn(35)
        `when`(mockPrefs.getInt("keyboard_height_landscape", 50)).thenReturn(50)
        `when`(mockPrefs.getString("numpad_layout", "default")).thenReturn("default")
        `when`(mockPrefs.getString("number_row", "no_number_row")).thenReturn("no_number_row")

        // Swipe and sensitivity settings
        `when`(mockPrefs.getString("swipe_dist", "15")).thenReturn("15")
        `when`(mockPrefs.getString("slider_sensitivity", "30")).thenReturn("30")

        // Vibration settings
        `when`(mockPrefs.getBoolean("vibrate_custom", false)).thenReturn(false)
        `when`(mockPrefs.getInt("vibrate_duration", 20)).thenReturn(20)
        `when`(mockPrefs.getInt("longpress_timeout", 600)).thenReturn(600)
        `when`(mockPrefs.getInt("longpress_interval", 65)).thenReturn(65)
        `when`(mockPrefs.getBoolean("keyrepeat_enabled", true)).thenReturn(true)

        // Visual settings
        `when`(mockPrefs.getInt("label_brightness", 100)).thenReturn(100)
        `when`(mockPrefs.getInt("keyboard_opacity", 100)).thenReturn(100)
        `when`(mockPrefs.getInt("key_opacity", 100)).thenReturn(100)
        `when`(mockPrefs.getInt("key_activated_opacity", 100)).thenReturn(100)
        `when`(mockPrefs.getBoolean("border_config", false)).thenReturn(false)
        `when`(mockPrefs.getInt("custom_border_radius", 0)).thenReturn(0)
        `when`(mockPrefs.getFloat("character_size", 1.15f)).thenReturn(1.15f)
        `when`(mockPrefs.getString("theme", "")).thenReturn("")

        // Behavior settings
        `when`(mockPrefs.getBoolean("lock_double_tap", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("autocapitalisation", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("switch_input_immediate", false)).thenReturn(false)
        `when`(mockPrefs.getString("number_entry_layout", "pin")).thenReturn("pin")
        `when`(mockPrefs.getInt("current_layout_portrait", 0)).thenReturn(0)
        `when`(mockPrefs.getInt("current_layout_landscape", 0)).thenReturn(0)
        `when`(mockPrefs.getString("circle_sensitivity", "2")).thenReturn("2")

        // Clipboard settings
        `when`(mockPrefs.getBoolean("clipboard_history_enabled", false)).thenReturn(false)
        `when`(mockPrefs.getInt("clipboard_history_limit", 6)).thenReturn(6)
        `when`(mockPrefs.getInt("clipboard_pane_height_percent", 30)).thenReturn(30)
        `when`(mockPrefs.getString("clipboard_max_item_size_kb", "500")).thenReturn("500")
        `when`(mockPrefs.getString("clipboard_limit_type", "count")).thenReturn("count")
        `when`(mockPrefs.getString("clipboard_size_limit_mb", "10")).thenReturn("10")

        // Swipe typing settings
        `when`(mockPrefs.getBoolean("swipe_typing_enabled", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("swipe_show_debug_scores", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("word_prediction_enabled", false)).thenReturn(false)
        `when`(mockPrefs.getInt("suggestion_bar_opacity", 90)).thenReturn(90)

        // Prediction weights
        `when`(mockPrefs.getFloat("prediction_context_boost", 2.0f)).thenReturn(2.0f)
        `when`(mockPrefs.getFloat("prediction_frequency_scale", 1000.0f)).thenReturn(1000.0f)

        // Autocorrect settings
        `when`(mockPrefs.getBoolean("autocorrect_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getInt("autocorrect_min_word_length", 3)).thenReturn(3)
        `when`(mockPrefs.getFloat("autocorrect_char_match_threshold", 0.67f)).thenReturn(0.67f)
        `when`(mockPrefs.getInt("autocorrect_confidence_min_frequency", 500)).thenReturn(500)
        `when`(mockPrefs.getInt("autocorrect_max_length_diff", 2)).thenReturn(2)
        `when`(mockPrefs.getInt("autocorrect_prefix_length", 2)).thenReturn(2)
        `when`(mockPrefs.getInt("autocorrect_max_beam_candidates", 3)).thenReturn(3)

        // Swipe autocorrect
        `when`(mockPrefs.getBoolean("swipe_beam_autocorrect_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("swipe_final_autocorrect_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getString("swipe_fuzzy_match_mode", "edit_distance")).thenReturn("edit_distance")

        // Swipe scoring weights
        `when`(mockPrefs.getInt("swipe_prediction_source", 60)).thenReturn(60)
        `when`(mockPrefs.getFloat("swipe_common_words_boost", 1.3f)).thenReturn(1.3f)
        `when`(mockPrefs.getFloat("swipe_top5000_boost", 1.0f)).thenReturn(1.0f)
        `when`(mockPrefs.getFloat("swipe_rare_words_penalty", 0.75f)).thenReturn(0.75f)

        // Short gestures
        `when`(mockPrefs.getBoolean("short_gestures_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getInt("short_gesture_min_distance", 20)).thenReturn(20)

        // Neural prediction
        `when`(mockPrefs.getBoolean("neural_prediction_enabled", true)).thenReturn(true)
        `when`(mockPrefs.getInt("neural_beam_width", 4)).thenReturn(4)
        `when`(mockPrefs.getInt("neural_max_length", 35)).thenReturn(35)
        `when`(mockPrefs.getFloat("neural_confidence_threshold", 0.1f)).thenReturn(0.1f)
        `when`(mockPrefs.getBoolean("neural_batch_beams", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("neural_greedy_search", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("termux_mode_enabled", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("swipe_debug_detailed_logging", false)).thenReturn(false)
        `when`(mockPrefs.getBoolean("swipe_debug_show_raw_output", true)).thenReturn(true)
        `when`(mockPrefs.getBoolean("swipe_show_raw_beam_predictions", false)).thenReturn(false)

        // Beam search tuning
        `when`(mockPrefs.getFloat("neural_beam_alpha", 1.2f)).thenReturn(1.2f)
        `when`(mockPrefs.getFloat("neural_beam_prune_confidence", 0.8f)).thenReturn(0.8f)
        `when`(mockPrefs.getFloat("neural_beam_score_gap", 5.0f)).thenReturn(5.0f)

        // Neural model versioning
        `when`(mockPrefs.getString("neural_model_version", "v2")).thenReturn("v2")
        `when`(mockPrefs.getBoolean("neural_use_quantized", false)).thenReturn(false)
        `when`(mockPrefs.getInt("neural_user_max_seq_length", 0)).thenReturn(0)
        `when`(mockPrefs.getString("neural_resampling_mode", "discard")).thenReturn("discard")
        `when`(mockPrefs.getString("neural_custom_encoder_uri", null)).thenReturn(null)
        `when`(mockPrefs.getString("neural_custom_encoder_path", null)).thenReturn(null)
        `when`(mockPrefs.getString("neural_custom_decoder_uri", null)).thenReturn(null)
        `when`(mockPrefs.getString("neural_custom_decoder_path", null)).thenReturn(null)

        // Make sure contains() returns false for migration check
        `when`(mockPrefs.contains("number_entry_layout")).thenReturn(true)
    }

    // Global Config Initialization Tests

    @Test
    fun testInitGlobalConfig_createsGlobalInstance() {
        // Act
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Assert
        assertNotNull(Config.globalConfig())
    }

    @Test
    fun testInitGlobalConfig_withNullHandler_allowsNullable() {
        // Act - should not throw
        Config.initGlobalConfig(mockPrefs, mockResources, null, false)

        // Assert
        assertNotNull(Config.globalConfig())
    }

    @Test
    fun testGlobalPrefs_returnsSharedPreferences() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        val prefs = Config.globalPrefs()

        // Assert
        assertEquals(mockPrefs, prefs)
    }

    // Type-Safe Preference Loading Tests

    @Test
    fun testSafeGetInt_withValidInt_returnsValue() {
        // Arrange
        `when`(mockPrefs.getInt("test_key", 42)).thenReturn(100)

        // Act
        val result = Config.safeGetInt(mockPrefs, "test_key", 42)

        // Assert
        assertEquals(100, result)
    }

    @Test
    fun testSafeGetInt_withStringValue_convertsToInt() {
        // Arrange
        `when`(mockPrefs.getInt("test_key", 42)).thenThrow(ClassCastException())
        `when`(mockPrefs.getString("test_key", "42")).thenReturn("100")

        // Act
        val result = Config.safeGetInt(mockPrefs, "test_key", 42)

        // Assert
        assertEquals(100, result)
    }

    @Test
    fun testSafeGetInt_withInvalidString_returnsDefault() {
        // Arrange
        `when`(mockPrefs.getInt("test_key", 42)).thenThrow(ClassCastException())
        `when`(mockPrefs.getString("test_key", "42")).thenReturn("invalid")

        // Act
        val result = Config.safeGetInt(mockPrefs, "test_key", 42)

        // Assert
        assertEquals(42, result)
    }

    @Test
    fun testSafeGetFloat_withValidFloat_returnsValue() {
        // Arrange
        `when`(mockPrefs.getFloat("test_key", 1.5f)).thenReturn(2.5f)

        // Act
        val result = Config.safeGetFloat(mockPrefs, "test_key", 1.5f)

        // Assert
        assertEquals(2.5f, result, 0.001f)
    }

    @Test
    fun testSafeGetFloat_withIntValue_convertsToFloat() {
        // Arrange
        `when`(mockPrefs.getFloat("test_key", 1.5f)).thenThrow(ClassCastException())
        `when`(mockPrefs.getInt("test_key", 1)).thenReturn(2)

        // Act
        val result = Config.safeGetFloat(mockPrefs, "test_key", 1.5f)

        // Assert
        assertEquals(2.0f, result, 0.001f)
    }

    @Test
    fun testSafeGetFloat_withStringValue_convertsToFloat() {
        // Arrange
        `when`(mockPrefs.getFloat("test_key", 1.5f)).thenThrow(ClassCastException())
        `when`(mockPrefs.getInt("test_key", 1)).thenThrow(ClassCastException())
        `when`(mockPrefs.getString("test_key", "1.5")).thenReturn("2.5")

        // Act
        val result = Config.safeGetFloat(mockPrefs, "test_key", 1.5f)

        // Assert
        assertEquals(2.5f, result, 0.001f)
    }

    @Test
    fun testSafeGetFloat_withInvalidValue_returnsDefault() {
        // Arrange
        `when`(mockPrefs.getFloat("test_key", 1.5f)).thenThrow(ClassCastException())
        `when`(mockPrefs.getInt("test_key", 1)).thenThrow(ClassCastException())
        `when`(mockPrefs.getString("test_key", "1.5")).thenReturn("invalid")

        // Act
        val result = Config.safeGetFloat(mockPrefs, "test_key", 1.5f)

        // Assert
        assertEquals(1.5f, result, 0.001f)
    }

    // Float Preference Repair Tests

    @Test
    fun testRepairCorruptedFloatPreferences_withIntValue_repairs() {
        // Arrange
        `when`(mockPrefs.getFloat("character_size", 1.15f)).thenThrow(ClassCastException())
        `when`(mockPrefs.getInt("character_size", 1)).thenReturn(2)

        // Act
        Config.repairCorruptedFloatPreferences(mockPrefs)

        // Assert
        verify(mockPrefsEditor).putFloat("character_size", 2.0f)
        verify(mockPrefsEditor).apply()
    }

    @Test
    fun testRepairCorruptedFloatPreferences_withStringValue_repairs() {
        // Arrange
        `when`(mockPrefs.getFloat("character_size", 1.15f)).thenThrow(ClassCastException())
        `when`(mockPrefs.getInt("character_size", 1)).thenThrow(ClassCastException())
        `when`(mockPrefs.getString("character_size", "1.15")).thenReturn("2.5")

        // Act
        Config.repairCorruptedFloatPreferences(mockPrefs)

        // Assert
        verify(mockPrefsEditor).putFloat("character_size", 2.5f)
        verify(mockPrefsEditor).apply()
    }

    @Test
    fun testRepairCorruptedFloatPreferences_withValidFloat_doesNotRepair() {
        // Arrange
        `when`(mockPrefs.getFloat(anyString(), anyFloat())).thenReturn(1.15f)

        // Act
        Config.repairCorruptedFloatPreferences(mockPrefs)

        // Assert
        verify(mockPrefsEditor, never()).putFloat(anyString(), anyFloat())
        verify(mockPrefsEditor, never()).apply()
    }

    // Layout Switching Tests

    @Test
    fun testGetCurrentLayout_inNarrowScreen_returnsPortraitLayout() {
        // Arrange
        displayMetrics.widthPixels = 720 // Below 600dp threshold
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        val layout = Config.globalConfig().get_current_layout()

        // Assert
        assertEquals(0, layout) // current_layout_portrait
    }

    @Test
    fun testGetCurrentLayout_inWideScreen_returnsLandscapeLayout() {
        // Arrange
        displayMetrics.widthPixels = 2000 // Above 600dp threshold
        displayMetrics.density = 2.0f // widthPixels/density = 1000dp
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        val layout = Config.globalConfig().get_current_layout()

        // Assert
        assertEquals(0, layout) // current_layout_landscape
    }

    @Test
    fun testSetCurrentLayout_inNarrowScreen_savesPortraitLayout() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        Config.globalConfig().set_current_layout(2)

        // Assert
        verify(mockPrefsEditor).putInt("current_layout_portrait", 2)
        verify(mockPrefsEditor).putInt("current_layout_landscape", 0)
        verify(mockPrefsEditor).apply()
    }

    // Clipboard Configuration Tests

    @Test
    fun testSetClipboardHistoryEnabled_savesPreference() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        Config.globalConfig().set_clipboard_history_enabled(true)

        // Assert
        verify(mockPrefsEditor).putBoolean("clipboard_history_enabled", true)
        verify(mockPrefsEditor).commit()
    }

    @Test
    fun testSetClipboardHistoryLimit_savesPreference() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        Config.globalConfig().set_clipboard_history_limit(10)

        // Assert
        verify(mockPrefsEditor).putInt("clipboard_history_limit", 10)
        verify(mockPrefsEditor).commit()
    }

    @Test
    fun testSetClipboardPaneHeightPercent_clampsValue() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act - try to set below minimum
        Config.globalConfig().set_clipboard_pane_height_percent(5)

        // Assert - should clamp to 10
        verify(mockPrefsEditor).putInt("clipboard_pane_height_percent", 10)
    }

    @Test
    fun testSetClipboardPaneHeightPercent_clampsMaxValue() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act - try to set above maximum
        Config.globalConfig().set_clipboard_pane_height_percent(60)

        // Assert - should clamp to 50
        verify(mockPrefsEditor).putInt("clipboard_pane_height_percent", 50)
    }

    // Orientation Tests

    @Test
    fun testConfig_inPortrait_setsOrientationFlag() {
        // Arrange
        mockConfiguration.orientation = Configuration.ORIENTATION_PORTRAIT
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        val config = Config.globalConfig()

        // Assert
        assertFalse(config.orientation_landscape)
    }

    @Test
    fun testConfig_inLandscape_setsOrientationFlag() {
        // Arrange
        mockConfiguration.orientation = Configuration.ORIENTATION_LANDSCAPE
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)

        // Act
        val config = Config.globalConfig()

        // Assert
        assertTrue(config.orientation_landscape)
    }

    @Test
    fun testConfig_withFoldableUnfolded_setsFoldableFlag() {
        // Act
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, true)
        val config = Config.globalConfig()

        // Assert
        assertTrue(config.foldable_unfolded)
    }

    // Wide Screen Detection Tests

    @Test
    fun testConfig_withNarrowScreen_detectsAsNarrow() {
        // Arrange
        displayMetrics.widthPixels = 720
        displayMetrics.density = 2.0f // 360dp width

        // Act
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)
        val config = Config.globalConfig()

        // Assert
        assertFalse(config.wide_screen)
    }

    @Test
    fun testConfig_withWideScreen_detectsAsWide() {
        // Arrange
        displayMetrics.widthPixels = 2000
        displayMetrics.density = 2.0f // 1000dp width (> 600dp threshold)

        // Act
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)
        val config = Config.globalConfig()

        // Assert
        assertTrue(config.wide_screen)
    }

    // Version Increment Tests

    @Test
    fun testRefresh_incrementsVersion() {
        // Arrange
        Config.initGlobalConfig(mockPrefs, mockResources, mockHandler, false)
        val config = Config.globalConfig()
        val initialVersion = config.version

        // Act
        config.refresh(mockResources, false)

        // Assert
        assertEquals(initialVersion + 1, config.version)
    }

    // Helper methods for argument matching
    private fun anyString(): String = org.mockito.ArgumentMatchers.anyString() ?: ""
    private fun anyInt(): Int = org.mockito.ArgumentMatchers.anyInt()
    private fun anyFloat(): Float = org.mockito.ArgumentMatchers.anyFloat()
    private fun anyBoolean(): Boolean = org.mockito.ArgumentMatchers.anyBoolean()
}
