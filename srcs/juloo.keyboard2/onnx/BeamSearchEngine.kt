package juloo.keyboard2.onnx

/**
 * Beam search data structures and configuration.
 *
 * Beam search is a heuristic search algorithm that explores the most promising
 * candidates at each step, maintaining a fixed number of hypotheses (beams).
 *
 * This module contains the core data structures used by the beam search algorithm:
 * - BeamState: Represents a single hypothesis during search
 * - BeamCandidate: Final result with word and confidence
 * - BeamSearchConfig: Algorithm parameters
 *
 * The actual beam search algorithm implementation remains in OnnxSwipePredictor.java
 * and will be migrated in a future refactoring phase due to its complexity (410 lines).
 *
 * Thread Safety: Data classes are immutable where possible and thread-safe.
 */

/**
 * Configuration parameters for beam search decoding.
 *
 * @param beamWidth Number of hypotheses to maintain at each step (e.g., 5)
 * @param maxLength Maximum sequence length to decode (e.g., 20)
 * @param decoderSeqLength Fixed decoder sequence length from model export (e.g., 20)
 * @param vocabSize Size of token vocabulary
 * @param confidenceThreshold Minimum confidence to accept predictions (e.g., 0.05)
 * @param useBatchedDecoding Whether to process beams in batches (faster but may have issues)
 */
data class BeamSearchConfig(
    val beamWidth: Int = 5,
    val maxLength: Int = 20,
    val decoderSeqLength: Int = 20,
    val vocabSize: Int,
    val confidenceThreshold: Float = 0.05f,
    val useBatchedDecoding: Boolean = false
) {
    init {
        require(beamWidth > 0) { "beamWidth must be positive" }
        require(maxLength > 0) { "maxLength must be positive" }
        require(decoderSeqLength > 0) { "decoderSeqLength must be positive" }
        require(vocabSize > 0) { "vocabSize must be positive" }
        require(confidenceThreshold >= 0f && confidenceThreshold <= 1f) {
            "confidenceThreshold must be between 0 and 1"
        }
    }
}

/**
 * Represents a single beam state during beam search.
 *
 * Each beam maintains:
 * - Token sequence generated so far
 * - Cumulative score (negative log-likelihood)
 * - Finished flag (true when EOS token generated)
 *
 * Thread Safety: Mutable for performance during search. Not thread-safe.
 */
data class BeamState(
    val tokens: MutableList<Long>,
    var score: Float,
    var finished: Boolean
) {
    /**
     * Create initial beam with start-of-sequence token.
     */
    constructor(startToken: Int, startScore: Float, isFinished: Boolean) : this(
        tokens = mutableListOf(startToken.toLong()),
        score = startScore,
        finished = isFinished
    )

    /**
     * Create copy of existing beam for branching.
     */
    constructor(other: BeamState) : this(
        tokens = other.tokens.toMutableList(),
        score = other.score,
        finished = other.finished
    )

    /**
     * Add token to this beam's sequence.
     */
    fun addToken(token: Long, logProb: Float) {
        tokens.add(token)
        score += logProb  // Accumulate negative log-likelihood
    }

    /**
     * Get length of token sequence.
     */
    val length: Int
        get() = tokens.size

    /**
     * Get last token in sequence.
     */
    val lastToken: Long?
        get() = tokens.lastOrNull()
}

/**
 * Final beam search candidate with decoded word and confidence.
 *
 * @param word Decoded word string
 * @param confidence Probability score [0, 1] computed as exp(-score)
 * @param tokens Original token sequence (for debugging)
 */
data class BeamCandidate(
    val word: String,
    val confidence: Float,
    val tokens: List<Long> = emptyList()
) : Comparable<BeamCandidate> {
    /**
     * Compare by confidence (descending).
     */
    override fun compareTo(other: BeamCandidate): Int {
        return other.confidence.compareTo(this.confidence)
    }

    /**
     * Check if this is a valid prediction.
     */
    fun isValid(minConfidence: Float, minLength: Int = 1): Boolean {
        return word.isNotEmpty() &&
                word.length >= minLength &&
                confidence >= minConfidence
    }
}

/**
 * Utility for top-K selection during beam search.
 *
 * Efficiently finds the K largest elements from a float array.
 */
object TopKSelector {
    /**
     * Index-value pair for tracking top-K elements.
     */
    data class IndexValue(val index: Int, val value: Float) : Comparable<IndexValue> {
        override fun compareTo(other: IndexValue): Int {
            return other.value.compareTo(this.value) // Descending order
        }
    }

    /**
     * Find top K indices with highest values using partial sort.
     *
     * More efficient than full sort for large arrays with small K.
     *
     * @param logits Probability array to search
     * @param k Number of top elements to find
     * @return List of (index, value) pairs in descending order by value
     */
    fun topK(logits: FloatArray, k: Int): List<IndexValue> {
        require(k > 0) { "k must be positive" }
        require(k <= logits.size) { "k cannot exceed array size" }

        // Create index-value pairs
        val pairs = logits.mapIndexed { index, value ->
            IndexValue(index, value)
        }

        // Partial sort: only sort top K elements
        return pairs.sortedDescending().take(k)
    }

    /**
     * Apply softmax to logits to get probabilities.
     *
     * @param logits Raw model output logits
     * @return Normalized probabilities that sum to 1.0
     */
    fun softmax(logits: FloatArray): FloatArray {
        // Numerical stability: subtract max value
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = logits.map { kotlin.math.exp(it - maxLogit) }
        val sumExp = expLogits.sum()
        return expLogits.map { (it / sumExp).toFloat() }.toFloatArray()
    }
}

/**
 * Token vocabulary constants.
 */
object TokenVocab {
    const val PAD_IDX = 0L      // Padding token
    const val UNK_IDX = 1L      // Unknown token
    const val SOS_IDX = 2L      // Start-of-sequence token
    const val EOS_IDX = 3L      // End-of-sequence token
    const val FIRST_CHAR = 4L   // First character token ('a')
    const val LAST_CHAR = 29L   // Last character token ('z')

    /**
     * Check if token is a special token (not a character).
     */
    fun isSpecialToken(token: Long): Boolean {
        return token < FIRST_CHAR
    }

    /**
     * Check if token is a character token.
     */
    fun isCharToken(token: Long): Boolean {
        return token in FIRST_CHAR..LAST_CHAR
    }

    /**
     * Convert character token to character.
     */
    fun tokenToChar(token: Long): Char? {
        return if (isCharToken(token)) {
            ('a' + (token - FIRST_CHAR).toInt())
        } else {
            null
        }
    }

    /**
     * Convert character to token.
     */
    fun charToToken(char: Char): Long? {
        return if (char in 'a'..'z') {
            FIRST_CHAR + (char - 'a')
        } else {
            null
        }
    }
}
