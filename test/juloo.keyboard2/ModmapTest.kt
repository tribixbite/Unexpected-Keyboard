package juloo.keyboard2

import org.junit.Test
import org.junit.Assert.*

class ModmapTest {

    @Test
    fun test() {
        val mm = Modmap()
        mm.add(Modmap.M.Shift, KeyValue.getKeyByName("a"), KeyValue.getKeyByName("b"))
        mm.add(Modmap.M.Fn, KeyValue.getKeyByName("c"), KeyValue.getKeyByName("d"))
        Utils.apply(mm, "a", KeyValue.Modifier.SHIFT, "b")
        Utils.apply(mm, "a", KeyValue.Modifier.FN, "æ")
        Utils.apply(mm, "c", KeyValue.Modifier.FN, "d")
    }

    @Test
    fun keyevent_mappings() {
        val mm = Modmap()
        mm.add(Modmap.M.Ctrl, KeyValue.getKeyByName("љ"), KeyValue.getKeyByName("љ:q"))
        Utils.apply(mm, "a", KeyValue.Modifier.CTRL, KeyValue.getKeyByName("a").withKeyevent(29))
        Utils.apply(mm, "љ", KeyValue.Modifier.CTRL, KeyValue.getKeyByName("љ").withKeyevent(45))
    }

    object Utils {
        fun apply(mm: Modmap, a: String, mod: KeyValue.Modifier, expected: String) {
            apply(mm, a, mod, KeyValue.getKeyByName(expected))
        }

        fun apply(mm: Modmap, a: String, mod: KeyValue.Modifier, expected: KeyValue) {
            KeyModifier.set_modmap(mm)
            val b = KeyModifier.modify(KeyValue.getKeyByName(a), mod)
            KeyModifier.set_modmap(null)
            assertEquals(b, expected)
        }
    }
}
