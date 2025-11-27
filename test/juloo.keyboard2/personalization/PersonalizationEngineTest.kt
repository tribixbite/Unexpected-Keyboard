package juloo.keyboard2.personalization

import android.content.Context
import android.content.SharedPreferences
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations

class PersonalizationEngineTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var engine: PersonalizationEngine

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)

        // Mock SharedPreferences for both PersonalizationEngine and UserVocabulary
        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockPrefs.getBoolean(anyString(), anyBoolean())).thenReturn(true)
        `when`(mockPrefs.getString(anyString(), anyString())).thenReturn("BALANCED")
        `when`(mockPrefs.getString(eq("vocabulary_data"), any())).thenReturn(null)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putBoolean(anyString(), anyBoolean())).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)
        `when`(mockEditor.clear()).thenReturn(mockEditor)

        engine = PersonalizationEngine(mockContext)
    }

    @Test
    fun testInitialization() {
        assertNotNull(engine)
        assertTrue(engine.isEnabled())
        assertEquals(PersonalizationEngine.LearningAggression.BALANCED, engine.getLearningAggression())
    }

    @Test
    fun testSetEnabled() {
        engine.setEnabled(false)
        assertFalse(engine.isEnabled())

        engine.setEnabled(true)
        assertTrue(engine.isEnabled())
    }

    @Test
    fun testSetLearningAggression() {
        engine.setLearningAggression(PersonalizationEngine.LearningAggression.CONSERVATIVE)
        assertEquals(PersonalizationEngine.LearningAggression.CONSERVATIVE, engine.getLearningAggression())

        engine.setLearningAggression(PersonalizationEngine.LearningAggression.AGGRESSIVE)
        assertEquals(PersonalizationEngine.LearningAggression.AGGRESSIVE, engine.getLearningAggression())
    }

    @Test
    fun testRecordWordTyped_WhenEnabled() {
        engine.setEnabled(true)
        engine.recordWordTyped("kotlin")

        assertTrue(engine.hasWord("kotlin"))
    }

    @Test
    fun testRecordWordTyped_WhenDisabled() {
        engine.setEnabled(false)
        engine.recordWordTyped("kotlin")

        assertFalse(engine.hasWord("kotlin"))
    }

    @Test
    fun testRecordWordTyped_IgnoreShortWords() {
        engine.recordWordTyped("a")
        engine.recordWordTyped("")

        assertFalse(engine.hasWord("a"))
        assertEquals(0, engine.getVocabularySize())
    }

    @Test
    fun testRecordWordsTyped() {
        val words = listOf("kotlin", "java", "python")
        engine.recordWordsTyped(words)

        assertEquals(3, engine.getVocabularySize())
        assertTrue(engine.hasWord("kotlin"))
        assertTrue(engine.hasWord("java"))
        assertTrue(engine.hasWord("python"))
    }

    @Test
    fun testGetPersonalizationBoost_WhenDisabled() {
        engine.recordWordTyped("kotlin")
        engine.setEnabled(false)

        val boost = engine.getPersonalizationBoost("kotlin")
        assertEquals(0.0f, boost, 0.01f)
    }

    @Test
    fun testGetPersonalizationBoost_UnknownWord() {
        val boost = engine.getPersonalizationBoost("unknown")
        assertEquals(0.0f, boost, 0.01f)
    }

    @Test
    fun testGetPersonalizationBoost_KnownWord() {
        repeat(10) { engine.recordWordTyped("kotlin") }

        val boost = engine.getPersonalizationBoost("kotlin")
        assertTrue(boost > 0.0f)
    }

    @Test
    fun testGetPersonalizationBoost_AggressionMultiplier() {
        repeat(10) { engine.recordWordTyped("test") }

        // Conservative: 0.5x multiplier
        engine.setLearningAggression(PersonalizationEngine.LearningAggression.CONSERVATIVE)
        val conservativeBoost = engine.getPersonalizationBoost("test")

        // Balanced: 1.0x multiplier
        engine.setLearningAggression(PersonalizationEngine.LearningAggression.BALANCED)
        val balancedBoost = engine.getPersonalizationBoost("test")

        // Aggressive: 1.5x multiplier
        engine.setLearningAggression(PersonalizationEngine.LearningAggression.AGGRESSIVE)
        val aggressiveBoost = engine.getPersonalizationBoost("test")

        // Verify multiplier relationship
        assertTrue(conservativeBoost < balancedBoost)
        assertTrue(balancedBoost < aggressiveBoost)
        assertEquals(conservativeBoost * 2.0f, balancedBoost, 0.01f)
        assertEquals(conservativeBoost * 3.0f, aggressiveBoost, 0.01f)
    }

    @Test
    fun testGetWordUsage() {
        repeat(5) { engine.recordWordTyped("kotlin") }

        val usage = engine.getWordUsage("kotlin")
        assertNotNull(usage)
        assertEquals("kotlin", usage!!.word)
        assertEquals(5, usage.usageCount)
    }

    @Test
    fun testGetTopWords() {
        repeat(10) { engine.recordWordTyped("frequent") }
        repeat(5) { engine.recordWordTyped("medium") }
        repeat(1) { engine.recordWordTyped("rare") }

        val topWords = engine.getTopWords(2)
        assertEquals(2, topWords.size)
        assertEquals("frequent", topWords[0].word)
        assertEquals("medium", topWords[1].word)
    }

    @Test
    fun testCleanupStaleWords() {
        val currentTime = System.currentTimeMillis()

        // Add old word
        engine.recordWordTyped("old", currentTime - (100 * 86400000L))

        // Add recent word
        engine.recordWordTyped("recent", currentTime - (1 * 86400000L))

        assertEquals(2, engine.getVocabularySize())

        val removed = engine.cleanupStaleWords()

        assertEquals(1, removed)
        assertEquals(1, engine.getVocabularySize())
        assertTrue(engine.hasWord("recent"))
        assertFalse(engine.hasWord("old"))
    }

    @Test
    fun testClearAllData() {
        engine.recordWordTyped("word1")
        engine.recordWordTyped("word2")

        assertEquals(2, engine.getVocabularySize())

        engine.clearAllData()

        assertEquals(0, engine.getVocabularySize())
    }

    @Test
    fun testExportImportData() {
        repeat(10) { engine.recordWordTyped("kotlin") }
        repeat(5) { engine.recordWordTyped("java") }

        val json = engine.exportData()
        assertNotNull(json)

        engine.clearAllData()
        assertEquals(0, engine.getVocabularySize())

        val imported = engine.importData(json)
        assertEquals(2, imported)
        assertTrue(engine.hasWord("kotlin"))
        assertTrue(engine.hasWord("java"))
    }

    @Test
    fun testGetStats() {
        repeat(10) { engine.recordWordTyped("frequent") }
        repeat(5) { engine.recordWordTyped("medium") }

        val stats = engine.getStats()

        assertTrue(stats.enabled)
        assertEquals(PersonalizationEngine.LearningAggression.BALANCED, stats.aggression)
        assertEquals(2, stats.vocabularySize)
        assertTrue(stats.averageUsageCount > 0.0)
        assertEquals("frequent", stats.mostUsedWord?.word)
    }

    @Test
    fun testExplainBoost_Disabled() {
        engine.setEnabled(false)
        val explanation = engine.explainBoost("kotlin")

        assertFalse(explanation.enabled)
        assertEquals(0.0f, explanation.finalBoost, 0.01f)
        assertTrue(explanation.explanation.contains("disabled"))
    }

    @Test
    fun testExplainBoost_UnknownWord() {
        val explanation = engine.explainBoost("unknown")

        assertTrue(explanation.enabled)
        assertFalse(explanation.inVocabulary)
        assertEquals(0.0f, explanation.finalBoost, 0.01f)
        assertTrue(explanation.explanation.contains("not in"))
    }

    @Test
    fun testExplainBoost_KnownWord() {
        repeat(50) { engine.recordWordTyped("kotlin") }

        val explanation = engine.explainBoost("kotlin")

        assertTrue(explanation.enabled)
        assertTrue(explanation.inVocabulary)
        assertEquals(50, explanation.usageCount)
        assertTrue(explanation.frequencyScore > 0.0f)
        assertTrue(explanation.recencyScore > 0.0f)
        assertTrue(explanation.finalBoost > 0.0f)
        assertTrue(explanation.explanation.contains("Used 50 times"))
    }

    @Test
    fun testAggressionMultipliers() {
        assertEquals(0.5f, PersonalizationEngine.LearningAggression.CONSERVATIVE.multiplier, 0.01f)
        assertEquals(1.0f, PersonalizationEngine.LearningAggression.BALANCED.multiplier, 0.01f)
        assertEquals(1.5f, PersonalizationEngine.LearningAggression.AGGRESSIVE.multiplier, 0.01f)
    }

    @Test
    fun testRealWorldScenario_FrequentlyTypedWord() {
        // Simulate typing "kotlin" 100 times over last week
        val currentTime = System.currentTimeMillis()

        for (i in 1..100) {
            val timestamp = currentTime - (i * 60000L) // Every minute
            engine.recordWordTyped("kotlin", timestamp)
        }

        val boost = engine.getPersonalizationBoost("kotlin", currentTime)

        // Should have strong boost (frequency ~3.0, recency ~1.0, balanced 1.0x)
        assertTrue(boost > 2.5f)
    }

    @Test
    fun testRealWorldScenario_RareOldWord() {
        val currentTime = System.currentTimeMillis()

        // Typed once, 60 days ago
        engine.recordWordTyped("rare", currentTime - (60 * 86400000L))

        val boost = engine.getPersonalizationBoost("rare", currentTime)

        // Should have weak boost (frequency ~1.0, recency ~0.3)
        assertTrue(boost < 1.0f)
    }

    @Test
    fun testRealWorldScenario_MediumRecentWord() {
        val currentTime = System.currentTimeMillis()

        // Typed 10 times, all in last 3 days
        for (i in 1..10) {
            engine.recordWordTyped("medium", currentTime - (i * 3600000L)) // Every hour
        }

        val boost = engine.getPersonalizationBoost("medium", currentTime)

        // Should have moderate boost (frequency ~2.0, recency ~1.0)
        assertTrue(boost > 1.5f && boost < 3.0f)
    }
}
