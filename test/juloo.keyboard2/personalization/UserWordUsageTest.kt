package juloo.keyboard2.personalization

import org.junit.Assert.*
import org.junit.Test

class UserWordUsageTest {

    @Test
    fun testNormalizeWord() {
        assertEquals("kotlin", UserWordUsage.normalizeWord("Kotlin"))
        assertEquals("kotlin", UserWordUsage.normalizeWord("KOTLIN"))
        assertEquals("kotlin", UserWordUsage.normalizeWord("  kotlin  "))
        assertEquals("test word", UserWordUsage.normalizeWord("Test Word"))
        assertEquals("", UserWordUsage.normalizeWord("   "))
    }

    @Test
    fun testCalculateFrequencyScore() {
        // Logarithmic scaling tests
        assertEquals(1.0f, UserWordUsage.calculateFrequencyScore(1), 0.01f)
        assertEquals(2.0f, UserWordUsage.calculateFrequencyScore(10), 0.1f)
        assertEquals(3.0f, UserWordUsage.calculateFrequencyScore(100), 0.1f)
        assertEquals(4.0f, UserWordUsage.calculateFrequencyScore(1000), 0.1f)

        // Edge cases
        assertEquals(0f, UserWordUsage.calculateFrequencyScore(0))
        assertTrue(UserWordUsage.calculateFrequencyScore(5) > UserWordUsage.calculateFrequencyScore(1))
        assertTrue(UserWordUsage.calculateFrequencyScore(50) > UserWordUsage.calculateFrequencyScore(5))
    }

    @Test
    fun testGetRecencyScore_Recent() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "test",
            usageCount = 10,
            lastUsed = currentTime - (2 * 86400000L) // 2 days ago
        )

        // Should get full score (1.0) within 7 days
        val recencyScore = usage.getRecencyScore(currentTime)
        assertEquals(1.0f, recencyScore, 0.01f)
    }

    @Test
    fun testGetRecencyScore_Medium() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "test",
            usageCount = 10,
            lastUsed = currentTime - (20 * 86400000L) // 20 days ago
        )

        // Should get partial score (0.5-1.0) within 7-30 days
        val recencyScore = usage.getRecencyScore(currentTime)
        assertTrue(recencyScore > 0.5f && recencyScore < 1.0f)
    }

    @Test
    fun testGetRecencyScore_Old() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "test",
            usageCount = 10,
            lastUsed = currentTime - (60 * 86400000L) // 60 days ago
        )

        // Should get low score (0.1-0.5) within 30-90 days
        val recencyScore = usage.getRecencyScore(currentTime)
        assertTrue(recencyScore > 0.1f && recencyScore < 0.5f)
    }

    @Test
    fun testGetRecencyScore_VeryOld() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "test",
            usageCount = 10,
            lastUsed = currentTime - (100 * 86400000L) // 100 days ago
        )

        // Should get zero score beyond 90 days
        val recencyScore = usage.getRecencyScore(currentTime)
        assertEquals(0.0f, recencyScore, 0.01f)
    }

    @Test
    fun testGetFrequencyScore() {
        val usage1 = UserWordUsage("test", usageCount = 1, lastUsed = 0L)
        val usage10 = UserWordUsage("test", usageCount = 10, lastUsed = 0L)
        val usage100 = UserWordUsage("test", usageCount = 100, lastUsed = 0L)

        assertTrue(usage1.getFrequencyScore() < usage10.getFrequencyScore())
        assertTrue(usage10.getFrequencyScore() < usage100.getFrequencyScore())
    }

    @Test
    fun testGetPersonalizationBoost_FrequentAndRecent() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "kotlin",
            usageCount = 100, // High frequency
            lastUsed = currentTime - (1 * 86400000L) // 1 day ago (recent)
        )

        val boost = usage.getPersonalizationBoost(currentTime)
        // Frequency ~3.0, Recency ~1.0 → boost ~3.0
        assertTrue(boost > 2.5f && boost < 4.0f)
    }

    @Test
    fun testGetPersonalizationBoost_FrequentButOld() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "kotlin",
            usageCount = 100, // High frequency
            lastUsed = currentTime - (60 * 86400000L) // 60 days ago (old)
        )

        val boost = usage.getPersonalizationBoost(currentTime)
        // Frequency ~3.0, Recency ~0.3 → boost ~0.9
        assertTrue(boost > 0.5f && boost < 1.5f)
    }

    @Test
    fun testGetPersonalizationBoost_RareButRecent() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "rare",
            usageCount = 2, // Low frequency
            lastUsed = currentTime - (1 * 86400000L) // 1 day ago (recent)
        )

        val boost = usage.getPersonalizationBoost(currentTime)
        // Frequency ~1.5, Recency ~1.0 → boost ~1.5
        assertTrue(boost > 1.0f && boost < 2.0f)
    }

    @Test
    fun testIsStale_VeryOld() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "old",
            usageCount = 10,
            lastUsed = currentTime - (100 * 86400000L) // 100 days ago
        )

        assertTrue(usage.isStale(currentTime))
    }

    @Test
    fun testIsStale_OneTimeOld() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "typo",
            usageCount = 1, // Only used once
            lastUsed = currentTime - (40 * 86400000L) // 40 days ago
        )

        // One-time word older than 30 days is stale
        assertTrue(usage.isStale(currentTime))
    }

    @Test
    fun testIsStale_FrequentAndRecent() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "active",
            usageCount = 50,
            lastUsed = currentTime - (10 * 86400000L) // 10 days ago
        )

        assertFalse(usage.isStale(currentTime))
    }

    @Test
    fun testIsStale_OneTimeRecent() {
        val currentTime = System.currentTimeMillis()
        val usage = UserWordUsage(
            word = "new",
            usageCount = 1,
            lastUsed = currentTime - (5 * 86400000L) // 5 days ago
        )

        // One-time word within 30 days is not stale yet
        assertFalse(usage.isStale(currentTime))
    }

    @Test
    fun testRecordNewUsage() {
        val usage = UserWordUsage(
            word = "test",
            usageCount = 5,
            lastUsed = 1000L,
            firstUsed = 100L
        )

        val updated = usage.recordNewUsage(2000L)

        assertEquals(6, updated.usageCount)
        assertEquals(2000L, updated.lastUsed)
        assertEquals(100L, updated.firstUsed) // First used should not change
        assertEquals("test", updated.word)
    }

    @Test
    fun testRecordNewUsage_MultipleUpdates() {
        var usage = UserWordUsage("word", 1, 1000L, 1000L)

        usage = usage.recordNewUsage(2000L)
        assertEquals(2, usage.usageCount)
        assertEquals(2000L, usage.lastUsed)

        usage = usage.recordNewUsage(3000L)
        assertEquals(3, usage.usageCount)
        assertEquals(3000L, usage.lastUsed)

        assertEquals(1000L, usage.firstUsed) // First used never changes
    }

    @Test
    fun testToString() {
        val usage = UserWordUsage("kotlin", 50, System.currentTimeMillis())
        val str = usage.toString()

        assertTrue(str.contains("kotlin"))
        assertTrue(str.contains("count=50"))
        assertTrue(str.contains("freqScore"))
        assertTrue(str.contains("recencyScore"))
    }

    @Test
    fun testRecencyDecayGradient() {
        val currentTime = System.currentTimeMillis()

        // Test smooth decay over time
        val day1 = UserWordUsage("test", 10, currentTime - (1 * 86400000L))
        val day7 = UserWordUsage("test", 10, currentTime - (7 * 86400000L))
        val day15 = UserWordUsage("test", 10, currentTime - (15 * 86400000L))
        val day30 = UserWordUsage("test", 10, currentTime - (30 * 86400000L))
        val day60 = UserWordUsage("test", 10, currentTime - (60 * 86400000L))
        val day90 = UserWordUsage("test", 10, currentTime - (90 * 86400000L))

        val score1 = day1.getRecencyScore(currentTime)
        val score7 = day7.getRecencyScore(currentTime)
        val score15 = day15.getRecencyScore(currentTime)
        val score30 = day30.getRecencyScore(currentTime)
        val score60 = day60.getRecencyScore(currentTime)
        val score90 = day90.getRecencyScore(currentTime)

        // Verify monotonic decrease
        assertTrue(score1 >= score7)
        assertTrue(score7 > score15)
        assertTrue(score15 > score30)
        assertTrue(score30 > score60)
        assertTrue(score60 > score90)

        // Verify specific thresholds
        assertEquals(1.0f, score1, 0.01f) // Full score within 7 days
        assertTrue(score30 < 1.0f && score30 > 0.5f) // Partial at 30 days
        assertTrue(score60 < 0.5f && score60 > 0.1f) // Low at 60 days
        assertEquals(0.0f, score90, 0.01f) // Zero at 90 days
    }

    @Test
    fun testFrequencyLogarithmicScaling() {
        // Verify logarithmic curve provides diminishing returns
        val diff1to10 = UserWordUsage.calculateFrequencyScore(10) -
                       UserWordUsage.calculateFrequencyScore(1)

        val diff10to100 = UserWordUsage.calculateFrequencyScore(100) -
                         UserWordUsage.calculateFrequencyScore(10)

        val diff100to1000 = UserWordUsage.calculateFrequencyScore(1000) -
                           UserWordUsage.calculateFrequencyScore(100)

        // Each order of magnitude gives same score increase
        assertEquals(diff1to10, diff10to100, 0.01f)
        assertEquals(diff10to100, diff100to1000, 0.01f)
    }
}
