/*
 * Copyright (C) 2025 The FlorisBoard Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dev.patrickgold.florisboard.ime.text.gestures

import android.content.Context
import androidx.collection.LruCache
import androidx.collection.SparseArrayCompat
import androidx.collection.set
import dev.patrickgold.florisboard.ime.core.Subtype
import dev.patrickgold.florisboard.ime.keyboard.KeyData
import dev.patrickgold.florisboard.ime.text.key.KeyCode
import dev.patrickgold.florisboard.ime.text.keyboard.TextKey
import dev.patrickgold.florisboard.nlpManager
import juloo.keyboard2.Config
import java.text.Normalizer
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

private fun TextKey.baseCode(): Int {
    return (data as? KeyData)?.code ?: KeyCode.UNSPECIFIED
}

/**
 * Classifies gestures by comparing them with an "ideal gesture".
 *
 * Check out Étienne Desticourt's excellent write up at https://github.com/AnySoftKeyboard/AnySoftKeyboard/pull/1870
 */
class StatisticalGlideTypingClassifier(context: Context) : GlideTypingClassifier {
    private val nlpManager by context.nlpManager()

    private val gesture = Gesture()
    private var keysByCharacter: SparseArrayCompat<TextKey> = SparseArrayCompat()
    private var words: List<String> = emptyList()
    private var keys: ArrayList<TextKey> = arrayListOf()
    private lateinit var pruner: Pruner
    private var wordDataSubtype: Subtype? = null
    private var layoutSubtype: Subtype? = null
    private var currentSubtype: Subtype? = null
    val ready: Boolean
        get() = currentSubtype == layoutSubtype && wordDataSubtype == layoutSubtype && wordDataSubtype != null
    private val prunerCache = LruCache<Subtype, Pruner>(PRUNER_CACHE_SIZE)

    /**
     * The minimum distance between points to be added to a gesture.
     */
    private var distanceThresholdSquared = 0

    companion object {
        /**
         * Describes the allowed length variance in a gesture. If a gesture is too long or too short, it is immediately
         * discarded to save cycles.
         */
        private const val PRUNING_LENGTH_THRESHOLD = 8.42

        /**
         * describes the number of points to sample a gesture at, i.e the resolution.
         */
        private const val SAMPLING_POINTS: Int = 200

        /**
         * Standard deviation of the distribution of distances between the shapes of two gestures
         * representing the same word. It's expressed for normalized gestures and is therefore
         * independent of the keyboard or key size.
         */
        private const val SHAPE_STD = 22.08f

        /**
         * Standard deviation of the distribution of distances between the locations of two gestures
         * representing the same word. It's expressed as a factor of key radius as it's applied to
         * un-normalized gestures and is therefore dependent on the size of the keys/keyboard.
         */
        private const val LOCATION_STD = 0.5109f

        private const val VELOCITY_STD = 1.0f

        /**
         * This is a very small cache that caches suggestions, so that they aren't recalculated e.g when releasing
         * a pointer when the suggestions were already calculated. Avoids a lot of micro pauses.
         */
        private const val SUGGESTION_CACHE_SIZE = 5

        /**
         * For multiple subtypes, the pruner is cached.
         */
        private const val PRUNER_CACHE_SIZE = 5
    }

    override fun addGesturePoint(position: GlideTypingGesture.Detector.Position) {
        if (!gesture.isEmpty) {
            val dx = gesture.getLastX() - position.x
            val dy = gesture.getLastY() - position.y

            if (dx * dx + dy * dy > distanceThresholdSquared) {
                gesture.addPoint(position.x, position.y, position.velocity)
            }
        } else {
            gesture.addPoint(position.x, position.y, position.velocity)
        }
    }

    override fun setLayout(keyViews: List<TextKey>, subtype: Subtype) {
        setWordData(subtype)
        // stop duplicate calls
        if (layoutSubtype == subtype && keys == keyViews) {
            return
        }

        // if only layout changed but not subtype
        val layoutChanged = layoutSubtype == subtype

        keysByCharacter.clear()
        keys.clear()
        keyViews.forEach {
            keysByCharacter[it.baseCode()] = it
            keys.add(it)
        }
        layoutSubtype = subtype
        distanceThresholdSquared = (keyViews.first().visibleBounds.width / 4).toInt()
        distanceThresholdSquared *= distanceThresholdSquared

        if (
            (wordDataSubtype == layoutSubtype)
            || layoutChanged // should force a re-initialize
        ) {
            initializePruner(layoutChanged)
        }
    }

    override fun setWordData(subtype: Subtype) {
        // stop duplicate calls..
        if (wordDataSubtype == subtype) {
            return
        }

        this.words = nlpManager.getListOfWords(subtype)

        this.wordDataSubtype = subtype
        if (wordDataSubtype == layoutSubtype) {
            initializePruner(false)
        }
    }

    /**
     * Exists because Pruner requires both word data and layout are initialized,
     * however we don't know what order they're initialized in.
     */
    private fun initializePruner(invalidateCache: Boolean) {
        val currentSubtype = this.layoutSubtype!!
        val cached = when {
            invalidateCache -> null
            else -> prunerCache.get(currentSubtype)
        }
        if (cached == null) {
            this.pruner = Pruner(PRUNING_LENGTH_THRESHOLD, this.words, keysByCharacter)
            prunerCache.put(currentSubtype, this.pruner)
        } else {
            this.pruner = cached
        }
        this.currentSubtype = currentSubtype
    }

    override fun initGestureFromPointerData(pointerData: GlideTypingGesture.Detector.PointerData) {
        for (position in pointerData.positions) {
            addGesturePoint(position)
        }
    }

    private val lruSuggestionCache = LruCache<Pair<Gesture, Int>, List<String>>(SUGGESTION_CACHE_SIZE)
    override fun getSuggestions(maxSuggestionCount: Int, gestureCompleted: Boolean): List<String> {
        return when (val cached = lruSuggestionCache.get(Pair(this.gesture, maxSuggestionCount))) {
            null -> {
                val suggestions = unCachedGetSuggestions(maxSuggestionCount)
                lruSuggestionCache.put(Pair(this.gesture.clone(), maxSuggestionCount), suggestions)

                suggestions
            }
            else -> {
                cached
            }
        }
    }

    private fun unCachedGetSuggestions(maxSuggestionCount: Int): List<String> {
        val candidates = arrayListOf<String>()
        val candidateWeights = arrayListOf<Float>()
        val key = keys.firstOrNull() ?: return listOf()
        val radius = min(key.visibleBounds.height, key.visibleBounds.width)
        var remainingWords = pruner.pruneByPath(gesture, this.keys, this.words)
        val userGesture = gesture.resample(SAMPLING_POINTS)
        val normalizedUserGesture: Gesture = userGesture.normalizeByBoxSide()
        remainingWords = pruner.pruneByShape(gesture, remainingWords, keysByCharacter, keys)

        for (i in remainingWords.indices) {
            val word = remainingWords[i]
            val idealGestures = Gesture.generateIdealGestures(word, keysByCharacter)

            for (idealGesture in idealGestures) {
                val wordGesture = idealGesture.resample(SAMPLING_POINTS)
                val normalizedGesture: Gesture = wordGesture.normalizeByBoxSide()
                val shapeDistance = calcShapeDistance(normalizedGesture, normalizedUserGesture)
                val locationDistance = calcLocationDistance(wordGesture, userGesture)
                val velocityDistance = calcVelocityDistance(wordGesture, userGesture)
                val shapeProbability = calcGaussianProbability(shapeDistance, 0.0f, SHAPE_STD)
                val locationProbability = calcGaussianProbability(locationDistance, 0.0f, LOCATION_STD * radius)
                val velocityProbability = calcGaussianProbability(velocityDistance, 0.0f, Config.globalConfig().swipe_velocity_std)
                val frequency = 255f * nlpManager.getFrequencyForWord(currentSubtype!!, word).toFloat()
                val confidence = shapeProbability.pow(Config.globalConfig().swipe_confidence_shape_weight) * locationProbability.pow(Config.globalConfig().swipe_confidence_location_weight) * frequency.pow(Config.globalConfig().swipe_confidence_frequency_weight) * velocityProbability.pow(Config.globalConfig().swipe_confidence_velocity_weight)

                var candidateDistanceSortedIndex = 0
                var duplicateIndex = Int.MAX_VALUE

                while (candidateDistanceSortedIndex < candidateWeights.size
                    && candidateWeights[candidateDistanceSortedIndex] <= confidence
                ) {
                    if (candidates[candidateDistanceSortedIndex].contentEquals(word)) duplicateIndex =
                        candidateDistanceSortedIndex
                    candidateDistanceSortedIndex++
                }
                if (candidateDistanceSortedIndex < maxSuggestionCount && candidateDistanceSortedIndex <= duplicateIndex) {
                    if (duplicateIndex < Int.MAX_VALUE) {
                        candidateWeights.removeAt(duplicateIndex)
                        candidates.removeAt(duplicateIndex)
                    }
                    candidateWeights.add(candidateDistanceSortedIndex, confidence)
                    candidates.add(candidateDistanceSortedIndex, word)
                    if (candidateWeights.size > maxSuggestionCount) {
                        candidateWeights.removeAt(maxSuggestionCount)
                        candidates.removeAt(maxSuggestionCount)
                    }
                }
            }
        }

        return candidates
    }

    override fun clear() {
        gesture.clear()
    }

    private fun calcLocationDistance(gesture1: Gesture, gesture2: Gesture): Float {
        var totalDistance = 0.0f
        for (i in 0 until SAMPLING_POINTS) {
            val x1 = gesture1.getX(i)
            val x2 = gesture2.getX(i)
            val y1 = gesture1.getY(i)
            val y2 = gesture2.getY(i)
            val distance = abs(x1 - x2) + abs(y1 - y2)
            totalDistance += distance
        }
        return totalDistance / SAMPLING_POINTS / 2
    }

    private fun calcGaussianProbability(value: Float, mean: Float, standardDeviation: Float): Float {
        val factor = 1.0 / (standardDeviation * sqrt(2 * PI))
        val exponent = ((value - mean) / standardDeviation).toDouble().pow(2.0)
        val probability = factor * exp(-1.0 / 2 * exponent)
        return probability.toFloat()
    }

    private fun calcShapeDistance(gesture1: Gesture, gesture2: Gesture): Float {
        var distance: Float
        var totalDistance = 0.0f
        for (i in 0 until SAMPLING_POINTS) {
            val x1 = gesture1.getX(i)
            val x2 = gesture2.getX(i)
            val y1 = gesture1.getY(i)
            val y2 = gesture2.getY(i)
            distance = Gesture.distance(x1, y1, x2, y2)
            totalDistance += distance
        }
        return totalDistance
    }

    data class Node(val key: TextKey, val pointIndex: Int, val gScore: Float = Float.MAX_VALUE, val fScore: Float = Float.MAX_VALUE, val cameFrom: Node? = null)
    data class Edge(val to: Node, val weight: Float)

    private fun calcVelocityDistance(gesture1: Gesture, gesture2: Gesture): Float {
        var totalDistance = 0.0f
        for (i in 0 until SAMPLING_POINTS) {
            val v1 = gesture1.getVelocity(i)
            val v2 = gesture2.getVelocity(i)
            val distance = abs(v1 - v2)
            totalDistance += distance
        }
        return totalDistance / SAMPLING_POINTS
    }

    class Pruner(
        /**
         * The length difference between a user gesture and a word gesture above which a word will
         * be pruned.
         */
        private val lengthThreshold: Double,
        words: List<String>,
        keysByCharacter: SparseArrayCompat<TextKey>,
    ) {

        /**
         * Finds the words whose start and end letter are closest to the start and end points of the
         * user gesture.
         *
         * @param userGesture The current user gesture.
         * @param keys The keys on the keyboard.
         * @return A list of likely words.
         */
        fun pruneByPath(
            userGesture: Gesture,
            keys: Iterable<TextKey>,
            words: List<String>
        ): ArrayList<String> {
            val remainingWords = ArrayList<String>()
            val keySequences = findPath(userGesture, keys)

            for (word in words) {
                for (keySequence in keySequences) {
                    // More strict checking: word must follow key sequence in order
                    var keyIndex = 0
                    var isMatch = true
                    
                    for (char in word) {
                        val charCode = getCodeForChar(char)
                        var found = false
                        
                        // Look for this character's key starting from current position
                        for (i in keyIndex until keySequence.size) {
                            if (keySequence[i] == charCode) {
                                keyIndex = i + 1
                                found = true
                                break
                            }
                        }
                        
                        if (!found) {
                            isMatch = false
                            break
                        }
                    }
                    
                    if (isMatch) {
                        remainingWords.add(word)
                        break // a word only needs to be added once
                    }
                }
            }

            return remainingWords
        }

        /**
         * Finds the words whose ideal gesture length is within a certain threshold of the user
         * gesture's length.
         *
         * @param userGesture The current user gesture.
         * @param words A list of words to consider.
         * @return A list of words that remained after pruning the input list by length.
         */
        fun pruneByShape(
            userGesture: Gesture,
            words: ArrayList<String>,
            keysByCharacter: SparseArrayCompat<TextKey>,
            keys: List<TextKey>,
        ): ArrayList<String> {
            val remainingWords = ArrayList<String>()

            val key = keys.firstOrNull() ?: return arrayListOf()
            val radius = min(key.visibleBounds.height, key.visibleBounds.width)
            val userLength = userGesture.getLength()
            val userTurningPoints = userGesture.countTurningPoints()
            for (word in words) {
                val idealGestures = Gesture.generateIdealGestures(word, keysByCharacter)
                for (idealGesture in idealGestures) {
                    val wordIdealLength = getCachedIdealLength(word, idealGesture)
                    val wordIdealTurningPoints = idealGesture.countTurningPoints()
                    if (abs(userLength - wordIdealLength) < lengthThreshold * radius && abs(userTurningPoints - wordIdealTurningPoints) <= 2) { // Allow a tolerance of 2 turning points
                        remainingWords.add(word)
                    }
                }
            }
            return remainingWords
        }

        private val cachedIdealLength = ConcurrentHashMap<String, Float>()
        private fun getCachedIdealLength(word: String, idealGesture: Gesture): Float {
            return cachedIdealLength.getOrPut(word) { idealGesture.getLength() }
        }

        companion object {
            private fun getFirstKeyLastKey(
                word: String,
                keysByCharacter: SparseArrayCompat<TextKey>,
            ): Pair<Int, Int>? {
                val firstLetter = word[0]
                val lastLetter = word[word.length - 1]
                val firstBaseChar = Normalizer.normalize(firstLetter.toString(), Normalizer.Form.NFD)[0]
                val lastBaseChar = Normalizer.normalize(lastLetter.toString(), Normalizer.Form.NFD)[0]
                return when {
                    keysByCharacter.indexOfKey(firstBaseChar.code) < 0 || keysByCharacter.indexOfKey(lastBaseChar.code) < 0 -> {
                        null
                    }
                    else -> {
                        val firstKey = keysByCharacter[firstBaseChar.code]
                        val lastKey = keysByCharacter[lastBaseChar.code]
                        if (firstKey != null && lastKey != null) {
                            firstKey.baseCode() to lastKey.baseCode()
                        } else {
                            null
                        }
                    }
                }
            }

            /**
             * Finds a chosen number of keys closest to a given point on the keyboard.
             *
             * @param x X coordinate of the point.
             * @param y Y coordinate of the point.
             * @param n The number of keys to return.
             * @param keys The keys of the keyboard.
             * @return A list of the n closest keys.
             */
            private fun findNClosestKeys(
                x: Float, y: Float, n: Int, keys: Iterable<TextKey>
            ): List<TextKey> {
                val keyDistances = HashMap<TextKey, Float>()
                for (key in keys) {
                    val visibleBoundsCenter = key.visibleBounds.center
                    val distance = Gesture.distance(
                        visibleBoundsCenter.x,
                        visibleBoundsCenter.y,
                        x,
                        y
                    )
                    keyDistances[key] = distance
                }

                return keyDistances.entries.sortedWith { c1, c2 -> c1.value.compareTo(c2.value) }.take(n)
                    .map { it.key }
            }

        private fun findPath(userGesture: Gesture, keys: Iterable<TextKey>): List<List<Int>> {
            val openSet = PriorityQueue<Node> { a, b -> a.fScore.compareTo(b.fScore) }
            val closedSet = HashSet<Node>()

            val startNodes = findNClosestKeys(userGesture.getFirstX(), userGesture.getFirstY(), 3, keys).map { Node(it, 0, 0f, 0f) }
            openSet.addAll(startNodes)

            val paths = ArrayList<List<Int>>()

            while (openSet.isNotEmpty()) {
                val current = openSet.poll()

                if (current.pointIndex == userGesture.size - 1) {
                    val path = ArrayList<Int>()
                    var temp: Node? = current
                    while (temp != null) {
                        path.add(temp.key.baseCode())
                        temp = temp.cameFrom
                    }
                    paths.add(path.reversed())
                    if (paths.size >= 10) { // limit to 10 paths
                        return paths
                    }
                    continue
                }

                closedSet.add(current)

                val nextPointIndex = current.pointIndex + 1
                val nextX = userGesture.getX(nextPointIndex)
                val nextY = userGesture.getY(nextPointIndex)

                val neighbors = findNClosestKeys(nextX, nextY, 3, keys).map { Node(it, nextPointIndex) }

                for (neighbor in neighbors) {
                    if (closedSet.contains(neighbor)) {
                        continue
                    }

                    val tentativeGScore = current.gScore + Gesture.distance(current.key.visibleBounds.centerX().toFloat(), current.key.visibleBounds.centerY().toFloat(), neighbor.key.visibleBounds.centerX().toFloat(), neighbor.key.visibleBounds.centerY().toFloat())

                    if (tentativeGScore < neighbor.gScore) {
                        val newNeighbor = neighbor.copy(gScore = tentativeGScore, fScore = tentativeGScore + Gesture.distance(neighbor.key.visibleBounds.centerX().toFloat(), neighbor.key.visibleBounds.centerY().toFloat(), userGesture.getLastX(), userGesture.getLastY()), cameFrom = current)
                        if (!openSet.contains(newNeighbor)) {
                            openSet.add(newNeighbor)
                        }
                    }
                }
            }

            return paths
        }
        }

        init {
        }
    }

    class Gesture(
        private val xs: FloatArray = FloatArray(MAX_SIZE),
        private val ys: FloatArray = FloatArray(MAX_SIZE),
        private val velocities: FloatArray = FloatArray(MAX_SIZE),
        private var size: Int = 0,
    ) {
        companion object {
            // TODO: Find out optimal max size
            private const val MAX_SIZE = 500

            fun generateIdealGestures(word: String, keysByCharacter: SparseArrayCompat<TextKey>): List<Gesture> {
                val idealGesture = Gesture()
                val idealGestureWithLoops = Gesture()
                var previousLetter = '\u0000'
                var hasLoops = false

                // Add points for each key
                for (c in word) {
                    val lc = Character.toLowerCase(c)
                    var key = keysByCharacter[lc.code]
                    if (key == null) {
                        // Try finding the base character instead, e.g., the "e" key instead of "é"
                        val baseCharacter: Char = Normalizer.normalize(lc.toString(), Normalizer.Form.NFD)[0]
                        key = keysByCharacter[baseCharacter.code]
                        if (key == null) {
                            continue
                        }
                    }
                    val visibleBoundsCenter = key.visibleBounds.center

                    // We adda little loop on  the key for duplicate letters
                    // so that we can differentiate words like pool and poll, lull and lul, etc...
                    if (previousLetter == lc) {
                        // bottom right
                        idealGestureWithLoops.addPoint(
                            visibleBoundsCenter.x + key.visibleBounds.width / 4.0f,
                            visibleBoundsCenter.y + key.visibleBounds.height / 4.0f,
                            0.0f
                        )
                        // top right
                        idealGestureWithLoops.addPoint(
                            visibleBoundsCenter.x + key.visibleBounds.width / 4.0f,
                            visibleBoundsCenter.y - key.visibleBounds.height / 4.0f,
                            0.0f
                        )
                        // top left
                        idealGestureWithLoops.addPoint(
                            visibleBoundsCenter.x - key.visibleBounds.width / 4.0f,
                            visibleBoundsCenter.y - key.visibleBounds.height / 4.0f,
                            0.0f
                        )
                        // bottom left
                        idealGestureWithLoops.addPoint(
                            visibleBoundsCenter.x - key.visibleBounds.width / 4.0f,
                            visibleBoundsCenter.y + key.visibleBounds.height / 4.0f,
                            0.0f
                        )
                        hasLoops = true

                        idealGesture.addPoint(
                            visibleBoundsCenter.x,
                            visibleBoundsCenter.y,
                            0.0f
                        )
                    } else {
                        idealGesture.addPoint(
                            visibleBoundsCenter.x,
                            visibleBoundsCenter.y,
                            0.0f
                        )
                        idealGestureWithLoops.addPoint(
                            visibleBoundsCenter.x,
                            visibleBoundsCenter.y,
                            0.0f
                        )
                    }
                    previousLetter = lc
                }
                return when (hasLoops) {
                    true -> listOf(idealGesture, idealGestureWithLoops)
                    false -> listOf(idealGesture)
                }
            }

            fun distance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
                return sqrt((x1 - x2).pow(2) + (y1 - y2).pow(2))
            }
        }

        val isEmpty: Boolean
            get() = size == 0

        fun addPoint(x: Float, y: Float, velocity: Float) {
            if (size >= MAX_SIZE) {
                return
            }
            xs[size] = x
            ys[size] = y
            velocities[size] = velocity
            size += 1
        }

        /**
         * Resamples the gesture into a new gesture with the chosen number of points by oversampling
         * it.
         *
         * @param numPoints The number of points that the new gesture will have. Must be superior to
         * the number of points in the current gesture.
         * @return An oversampled copy of the gesture.
         */
        fun resample(numPoints: Int): Gesture {
            val interpointDistance = (getLength() / numPoints)
            val resampledGesture = Gesture()
            resampledGesture.addPoint(xs[0], ys[0], velocities[0])
            var lastX = xs[0]
            var lastY = ys[0]
            var lastVelocity = velocities[0]
            var newX: Float
            var newY: Float
            var cumulativeError = 0.0f

            // otherwise nothing happens if size is only 1:
            if (this.size == 1) {
                for (i in 0 until SAMPLING_POINTS) {
                    resampledGesture.addPoint(xs[0], ys[0], velocities[0])
                }
            }

            for (i in 0 until size - 1) {
                // We calculate the unit vector from the two points we're between in the actual
                // gesture
                var dx = xs[i + 1] - xs[i]
                var dy = ys[i + 1] - ys[i]
                val norm = sqrt(dx.pow(2.0f) + dy.pow(2.0f))
                dx /= norm
                dy /= norm

                // The number of evenly sampled points that fit between the two actual points
                var numNewPoints = norm / interpointDistance

                // The number of point that'd fit between the two actual points is often not round,
                // which means we'll get an increasingly large error as we resample the gesture
                // and round down that number. To compensate for this we keep track of the error
                // and add additional points when it gets too large.
                cumulativeError += numNewPoints - numNewPoints.toInt()
                if (cumulativeError > 1) {
                    numNewPoints = (numNewPoints.toInt() + cumulativeError.toInt()).toFloat()
                    cumulativeError %= 1
                }
                for (j in 0 until numNewPoints.toInt()) {
                    newX = lastX + dx * interpointDistance
                    newY = lastY + dy * interpointDistance
                    val newVelocity = lastVelocity + (velocities[i+1] - lastVelocity) * (j+1) / numNewPoints
                    lastX = newX
                    lastY = newY
                    lastVelocity = newVelocity
                    resampledGesture.addPoint(newX, newY, newVelocity)
                }
            }
            return resampledGesture
        }

        fun normalizeByBoxSide(): Gesture {
            val normalizedGesture = Gesture()

            var maxX = -1.0f
            var maxY = -1.0f
            var minX = 10000.0f
            var minY = 10000.0f

            for (i in 0 until size) {
                maxX = max(xs[i], maxX)
                maxY = max(ys[i], maxY)
                minX = min(xs[i], minX)
                minY = min(ys[i], minY)
            }

            val width = maxX - minX
            val height = maxY - minY
            val longestSide = max(max(width, height), 0.00001f)

            val centroidX = (width / 2 + minX) / longestSide
            val centroidY = (height / 2 + minY) / longestSide

            for (i in 0 until size) {
                val x = xs[i] / longestSide - centroidX
                val y = ys[i] / longestSide - centroidY
                normalizedGesture.addPoint(x, y, velocities[i])
            }

            return normalizedGesture
        }

        fun getFirstX(): Float = xs.getOrElse(0) { 0f }
        fun getFirstY(): Float = ys.getOrElse(0) { 0f }
        fun getLastX(): Float = xs.getOrElse(size - 1) { 0f }
        fun getLastY(): Float = ys.getOrElse(size - 1) { 0f }

        fun getLength(): Float {
            var length = 0f
            for (i in 1 until size) {
                val previousX = xs[i - 1]
                val previousY = ys[i - 1]
                val currentX = xs[i]
                val currentY = ys[i]
                length += distance(previousX, previousY, currentX, currentY)
            }

            return length
        }

        fun clear() {
            this.size = 0
        }

        fun getX(i: Int): Float = xs.getOrElse(i) { 0f }
        fun getY(i: Int): Float = ys.getOrElse(i) { 0f }
        fun getVelocity(i: Int): Float = velocities.getOrElse(i) { 0f }

        fun countTurningPoints(): Int {
            if (size < 3) {
                return 0
            }

            var turningPoints = 0
            for (i in 1 until size - 1) {
                val x1 = xs[i - 1]
                val y1 = ys[i - 1]
                val x2 = xs[i]
                val y2 = ys[i]
                val x3 = xs[i + 1]
                val y3 = ys[i + 1]

                val angle1 = kotlin.math.atan2(y2 - y1, x2 - x1)
                val angle2 = kotlin.math.atan2(y3 - y2, x3 - x2)
                var angleDiff = Math.toDegrees(angle2 - angle1.toDouble()).toFloat()
                if (angleDiff > 180) {
                    angleDiff -= 360
                } else if (angleDiff < -180) {
                    angleDiff += 360
                }

                if (abs(angleDiff) > Config.globalConfig().swipe_turning_point_threshold) { // 45 degrees threshold
                    turningPoints++
                }
            }
            return turningPoints
        }

        fun clone(): Gesture {
            return Gesture(xs.clone(), ys.clone(), velocities.clone(), size)
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as Gesture

            if (this.size != other.size) return false

            for (i in 0 until size) {
                if (xs[i] != other.xs[i] || ys[i] != other.ys[i]) return false
            }

            return true
        }

        override fun hashCode(): Int {
            var result = xs.contentHashCode()
            result = 31 * result + ys.contentHashCode()
            result = 31 * result + size
            return result
        }
    }
}
