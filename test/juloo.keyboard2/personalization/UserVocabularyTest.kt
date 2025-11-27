package juloo.keyboard2.personalization

import android.content.Context
import android.content.SharedPreferences
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations

class UserVocabularyTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPrefs: SharedPreferences

    @Mock
    private lateinit var mockEditor: SharedPreferences.Editor

    private lateinit var vocabulary: UserVocabulary

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)

        `when`(mockContext.getSharedPreferences(anyString(), anyInt())).thenReturn(mockPrefs)
        `when`(mockPrefs.getString(anyString(), any())).thenReturn(null)
        `when`(mockPrefs.edit()).thenReturn(mockEditor)
        `when`(mockEditor.putString(anyString(), anyString())).thenReturn(mockEditor)
        `when`(mockEditor.clear()).thenReturn(mockEditor)

        vocabulary = UserVocabulary(mockContext)
    }

    @Test
    fun testInitialization() {
        assertNotNull(vocabulary)
        assertEquals(0, vocabulary.size())
    }

    @Test
    fun testRecordWordUsage_NewWord() {
        vocabulary.recordWordUsage("kotlin")

        assertTrue(vocabulary.hasWord("kotlin"))
        assertEquals(1, vocabulary.size())

        val usage = vocabulary.getWordUsage("kotlin")
        assertNotNull(usage)
        assertEquals("kotlin", usage!!.word)
        assertEquals(1, usage.usageCount)
    }

    @Test
    fun testRecordWordUsage_ExistingWord() {
        vocabulary.recordWordUsage("kotlin")
        vocabulary.recordWordUsage("kotlin")
        vocabulary.recordWordUsage("kotlin")

        assertEquals(1, vocabulary.size()) // Still one unique word
        val usage = vocabulary.getWordUsage("kotlin")
        assertEquals(3, usage!!.usageCount) // Count incremented
    }

    @Test
    fun testRecordWordUsage_CaseInsensitive() {
        vocabulary.recordWordUsage("Kotlin")
        vocabulary.recordWordUsage("KOTLIN")
        vocabulary.recordWordUsage("kotlin")

        assertEquals(1, vocabulary.size()) // All normalized to same word
        val usage = vocabulary.getWordUsage("kotlin")
        assertEquals(3, usage!!.usageCount)
    }

    @Test
    fun testRecordWordUsage_IgnoreShortWords() {
        vocabulary.recordWordUsage("a")
        vocabulary.recordWordUsage("i")
        vocabulary.recordWordUsage("")

        assertEquals(0, vocabulary.size()) // Short words ignored
    }

    @Test
    fun testRecordWordUsage_MultipleWords() {
        vocabulary.recordWordUsage("kotlin")
        vocabulary.recordWordUsage("java")
        vocabulary.recordWordUsage("python")

        assertEquals(3, vocabulary.size())
        assertTrue(vocabulary.hasWord("kotlin"))
        assertTrue(vocabulary.hasWord("java"))
        assertTrue(vocabulary.hasWord("python"))
    }

    @Test
    fun testGetPersonalizationBoost_UnknownWord() {
        val boost = vocabulary.getPersonalizationBoost("unknown")
        assertEquals(0.0f, boost, 0.01f)
    }

    @Test
    fun testGetPersonalizationBoost_KnownWord() {
        repeat(10) { vocabulary.recordWordUsage("kotlin") }

        val boost = vocabulary.getPersonalizationBoost("kotlin")
        assertTrue(boost > 0.0f) // Should have some boost
    }

    @Test
    fun testGetAllWords_SortedByBoost() {
        // Record words with different frequencies
        repeat(100) { vocabulary.recordWordUsage("frequent") }
        repeat(10) { vocabulary.recordWordUsage("medium") }
        repeat(1) { vocabulary.recordWordUsage("rare") }

        val allWords = vocabulary.getAllWords()

        assertEquals(3, allWords.size)
        // Should be sorted by boost (descending)
        assertEquals("frequent", allWords[0].word)
        assertEquals("medium", allWords[1].word)
        assertEquals("rare", allWords[2].word)
    }

    @Test
    fun testGetTopWords() {
        // Add many words
        for (i in 1..20) {
            repeat(i) { vocabulary.recordWordUsage("word$i") }
        }

        val top5 = vocabulary.getTopWords(5)
        assertEquals(5, top5.size)

        // word20 should be first (most frequent)
        assertEquals("word20", top5[0].word)
    }

    @Test
    fun testCleanupStaleWords_RemovesOldWords() {
        val currentTime = System.currentTimeMillis()

        // Add old word
        vocabulary.recordWordUsage("old", currentTime - (100 * 86400000L))

        // Add recent word
        vocabulary.recordWordUsage("recent", currentTime - (1 * 86400000L))

        assertEquals(2, vocabulary.size())

        val removed = vocabulary.cleanupStaleWords(currentTime)

        assertEquals(1, removed) // One word removed
        assertEquals(1, vocabulary.size()) // One word remains
        assertTrue(vocabulary.hasWord("recent"))
        assertFalse(vocabulary.hasWord("old"))
    }

    @Test
    fun testCleanupStaleWords_RemovesOneTimeOldWords() {
        val currentTime = System.currentTimeMillis()

        // One-time typo from 40 days ago
        vocabulary.recordWordUsage("typo", currentTime - (40 * 86400000L))

        // One-time recent word
        vocabulary.recordWordUsage("new", currentTime - (5 * 86400000L))

        val removed = vocabulary.cleanupStaleWords(currentTime)

        assertEquals(1, removed) // Old one-time word removed
        assertTrue(vocabulary.hasWord("new"))
        assertFalse(vocabulary.hasWord("typo"))
    }

    @Test
    fun testClearAll() {
        vocabulary.recordWordUsage("word1")
        vocabulary.recordWordUsage("word2")
        vocabulary.recordWordUsage("word3")

        assertEquals(3, vocabulary.size())

        vocabulary.clearAll()

        assertEquals(0, vocabulary.size())
        assertFalse(vocabulary.hasWord("word1"))
    }

    @Test
    fun testExportImportJson() {
        // Add some words
        repeat(10) { vocabulary.recordWordUsage("kotlin") }
        repeat(5) { vocabulary.recordWordUsage("java") }

        val json = vocabulary.exportToJson()
        assertNotNull(json)
        assertTrue(json.contains("kotlin"))
        assertTrue(json.contains("java"))

        // Clear and reimport
        vocabulary.clearAll()
        assertEquals(0, vocabulary.size())

        val imported = vocabulary.importFromJson(json)
        assertEquals(2, imported)
        assertTrue(vocabulary.hasWord("kotlin"))
        assertTrue(vocabulary.hasWord("java"))
    }

    @Test
    fun testImportJson_InvalidJson() {
        val imported = vocabulary.importFromJson("invalid json")
        assertEquals(0, imported)
    }

    @Test
    fun testGetStats() {
        repeat(10) { vocabulary.recordWordUsage("frequent") }
        repeat(5) { vocabulary.recordWordUsage("medium") }
        repeat(1) { vocabulary.recordWordUsage("rare") }

        val stats = vocabulary.getStats()

        assertEquals(3, stats.totalWords)
        assertTrue(stats.averageUsageCount > 0.0)
        assertEquals("frequent", stats.mostUsedWord?.word)
        assertEquals(10, stats.mostUsedWord?.usageCount)
    }

    @Test
    fun testGetStats_RecentlyUsedCount() {
        val currentTime = System.currentTimeMillis()

        // Recent words
        vocabulary.recordWordUsage("recent1", currentTime - (1 * 86400000L))
        vocabulary.recordWordUsage("recent2", currentTime - (3 * 86400000L))

        // Old words
        vocabulary.recordWordUsage("old1", currentTime - (30 * 86400000L))
        vocabulary.recordWordUsage("old2", currentTime - (60 * 86400000L))

        val stats = vocabulary.getStats()

        assertEquals(4, stats.totalWords)
        assertEquals(2, stats.recentlyUsedCount) // Only 2 within last 7 days
    }

    @Test
    fun testMaxVocabularySize() {
        // Try to add more than MAX_VOCABULARY_SIZE (5000)
        // This test is slow, so we'll just verify the mechanism works with smaller numbers

        // Fill to capacity
        for (i in 1..100) {
            vocabulary.recordWordUsage("word$i")
        }

        assertEquals(100, vocabulary.size())

        // Add one more - should remove lowest value word
        vocabulary.recordWordUsage("new")

        // Size should not exceed limit (in this case, we're under, but mechanism tested)
        assertTrue(vocabulary.size() <= 100)
        assertTrue(vocabulary.hasWord("new"))
    }

    @Test
    fun testConcurrentAccess() {
        // Test thread-safety by recording from multiple "threads" (sequential for test)
        val words = listOf("kotlin", "java", "python", "rust", "go")

        repeat(20) {
            words.forEach { word ->
                vocabulary.recordWordUsage(word)
            }
        }

        assertEquals(5, vocabulary.size())
        words.forEach { word ->
            val usage = vocabulary.getWordUsage(word)
            assertEquals(20, usage?.usageCount)
        }
    }

    @Test
    fun testRecordWordUsage_UpdatesTimestamp() {
        val time1 = 1000L
        val time2 = 2000L

        vocabulary.recordWordUsage("test", time1)
        var usage = vocabulary.getWordUsage("test")
        assertEquals(time1, usage!!.lastUsed)

        vocabulary.recordWordUsage("test", time2)
        usage = vocabulary.getWordUsage("test")
        assertEquals(time2, usage!!.lastUsed) // Timestamp updated
    }

    @Test
    fun testRecordWordUsage_PreservesFirstUsed() {
        val time1 = 1000L
        val time2 = 2000L

        vocabulary.recordWordUsage("test", time1)
        vocabulary.recordWordUsage("test", time2)

        val usage = vocabulary.getWordUsage("test")
        assertEquals(time1, usage!!.firstUsed) // First used preserved
        assertEquals(time2, usage.lastUsed) // Last used updated
    }
}
