package juloo.keyboard2

import org.junit.Test
import org.junit.Assert.*

class ComposeKeyTest {

    @Test
    fun composeEquals() {
        // From Compose.pre
        assertEquals(apply("'e"), KeyValue.makeStringKey("é"))
        assertEquals(apply("e'"), KeyValue.makeStringKey("é"))
        // From extra.json
        assertEquals(apply("Vc"), KeyValue.makeStringKey("Č"))
        assertEquals(apply("\\n"), KeyValue.getKeyByName("\\n"))
        // From arabic.json
        assertEquals(apply("اا"), KeyValue.getKeyByName("combining_alef_above"))
        assertEquals(apply("ل۷"), KeyValue.makeStringKey("ڵ"))
        assertEquals(apply("۷ل"), KeyValue.makeStringKey("ڵ"))
        // From cyrillic.json
        assertEquals(apply(",г"), KeyValue.makeStringKey("ӻ"))
        assertEquals(apply("г,"), KeyValue.makeStringKey("ӻ"))
        assertEquals(apply("ач"), KeyValue.getKeyByName("combining_aigu"))
    }

    @Test
    fun fnEquals() {
        val state = ComposeKeyData.fn
        assertEquals(apply("<", state), KeyValue.makeStringKey("«"))
        assertEquals(apply("{", state), KeyValue.makeStringKey("‹"))
        // Named key
        assertEquals(apply("1", state), KeyValue.getKeyByName("f1"))
        assertEquals(apply(" ", state), KeyValue.getKeyByName("nbsp"))
        // Named 1-char key
        assertEquals(apply("ய", state), KeyValue.makeStringKey("௰", KeyValue.FLAG_SMALLER_FONT))
    }

    @Test
    fun stringKeys() {
        val state = ComposeKeyData.shift
        assertEquals(apply("𝕨", state), KeyValue.makeStringKey("𝕎"))
        assertEquals(apply("𝕩", state), KeyValue.makeStringKey("𝕏"))
    }

    private fun apply(seq: String): KeyValue {
        return ComposeKey.apply(ComposeKeyData.compose, seq)
    }

    private fun apply(seq: String, state: Int): KeyValue {
        return ComposeKey.apply(state, seq)
    }
}
