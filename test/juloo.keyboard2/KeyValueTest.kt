package juloo.keyboard2

import android.view.KeyEvent
import org.junit.Test
import org.junit.Assert.*

class KeyValueTest {

    @Test
    fun equals() {
        assertEquals(
            KeyValue.makeStringKey("Foo").withSymbol("Symbol"),
            KeyValue.makeMacro("Symbol", arrayOf(KeyValue.makeStringKey("Foo")), 0)
        )
        assertEquals(
            KeyValue.getSpecialKeyByName("tab"),
            KeyValue.keyeventKey(0xE00F, KeyEvent.KEYCODE_TAB, KeyValue.FLAG_KEY_FONT or KeyValue.FLAG_SMALLER_FONT)
        )
        assertEquals(
            KeyValue.getSpecialKeyByName("tab").withSymbol("t"),
            KeyValue.keyeventKey("t", KeyEvent.KEYCODE_TAB, 0)
        )
        assertEquals(
            KeyValue.getSpecialKeyByName("tab").withSymbol("tab"),
            KeyValue.keyeventKey("tab", KeyEvent.KEYCODE_TAB, KeyValue.FLAG_SMALLER_FONT)
        )
    }

    @Test
    fun numpad_script() {
        assertEquals(apply_numpad_script("hindu-arabic"), "٠١٢٣٤٥٦٧٨٩")
        assertEquals(apply_numpad_script("bengali"), "০১২৩৪৫৬৭৮৯")
        assertEquals(apply_numpad_script("devanagari"), "०१२३४५६७८९")
        assertEquals(apply_numpad_script("persian"), "۰۱۲۳۴۵۶۷۸۹")
        assertEquals(apply_numpad_script("gujarati"), "૦૧૨૩૪૫૬૭૮૯")
        assertEquals(apply_numpad_script("kannada"), "೦೧೨೩೪೫೬೭೮೯")
        assertEquals(apply_numpad_script("tamil"), "௦௧௨௩௪௫௬௭௮௯")
    }

    private fun apply_numpad_script(script: String): String {
        val b = StringBuilder()
        val map = KeyModifier.modify_numpad_script(script)
        for (c in "0123456789".toCharArray()) {
            b.append(ComposeKey.apply(map, c).char)
        }
        return b.toString()
    }
}
