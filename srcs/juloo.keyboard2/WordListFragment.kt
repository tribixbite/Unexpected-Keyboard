package juloo.keyboard2

import android.app.AlertDialog
import android.os.Bundle
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.launch

/**
 * Fragment displaying a list of words
 */
class WordListFragment : Fragment() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var emptyText: TextView
    private lateinit var loadingProgress: ProgressBar
    private lateinit var dataSource: DictionaryDataSource
    private lateinit var adapter: BaseWordAdapter

    private var tabType: TabType = TabType.ACTIVE

    enum class TabType {
        ACTIVE, DISABLED, USER, CUSTOM
    }

    companion object {
        private const val ARG_TAB_TYPE = "tab_type"

        fun newInstance(tabType: TabType): WordListFragment {
            val fragment = WordListFragment()
            val args = Bundle()
            args.putInt(ARG_TAB_TYPE, tabType.ordinal)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tabType = TabType.values()[arguments?.getInt(ARG_TAB_TYPE) ?: 0]
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_word_list, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        recyclerView = view.findViewById(R.id.recycler_view)
        emptyText = view.findViewById(R.id.empty_text)
        loadingProgress = view.findViewById(R.id.loading_progress)

        recyclerView.layoutManager = LinearLayoutManager(requireContext())

        initializeDataSource()
        setupAdapter()
        loadWords()
    }

    private fun initializeDataSource() {
        val prefs = DirectBootAwarePreferences.get_shared_preferences(requireContext())
        val disabledSource = DisabledDictionarySource(prefs)

        dataSource = when (tabType) {
            TabType.ACTIVE -> MainDictionarySource(requireContext(), disabledSource)
            TabType.DISABLED -> disabledSource
            TabType.USER -> UserDictionarySource(requireContext(), requireContext().contentResolver)
            TabType.CUSTOM -> CustomDictionarySource(prefs)
        }
    }

    private fun setupAdapter() {
        adapter = when (tabType) {
            TabType.CUSTOM -> {
                WordEditableAdapter(
                    onEdit = { word -> showEditDialog(word) },
                    onDelete = { word -> deleteWord(word) },
                    onAdd = { showAddDialog() }
                )
            }
            else -> {
                WordToggleAdapter { word, enabled ->
                    toggleWord(word, enabled)
                }
            }
        }

        recyclerView.adapter = adapter
    }

    private fun loadWords() {
        loadingProgress.visibility = View.VISIBLE
        emptyText.visibility = View.GONE

        lifecycleScope.launch {
            try {
                val words = dataSource.getAllWords()
                adapter.setWords(words)
                updateEmptyState()
            } catch (e: Exception) {
                emptyText.text = "Error loading words: ${e.message}"
                emptyText.visibility = View.VISIBLE
            } finally {
                loadingProgress.visibility = View.GONE
            }
        }
    }

    private var currentSourceFilter: WordSource? = null

    fun filter(query: String, sourceFilter: WordSource? = null) {
        if (!::adapter.isInitialized) return
        currentSourceFilter = sourceFilter
        adapter.filter(query, sourceFilter)
        updateEmptyState()
    }

    fun getFilteredCount(): Int {
        if (!::adapter.isInitialized) return 0
        return adapter.getFilteredCount()
    }

    private fun updateEmptyState() {
        if (!::adapter.isInitialized) return
        if (adapter.getFilteredCount() == 0) {
            emptyText.visibility = View.VISIBLE
            recyclerView.visibility = View.GONE
        } else {
            emptyText.visibility = View.GONE
            recyclerView.visibility = View.VISIBLE
        }
    }

    private fun toggleWord(word: DictionaryWord, enabled: Boolean) {
        lifecycleScope.launch {
            try {
                dataSource.toggleWord(word.word, enabled)
                loadWords()  // Reload to reflect changes
                // Notify parent activity to refresh other tabs
                (activity as? DictionaryManagerActivity)?.refreshAllTabs()
            } catch (e: Exception) {
                // Show error
                AlertDialog.Builder(requireContext())
                    .setTitle("Error")
                    .setMessage("Failed to toggle word: ${e.message}")
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    private fun deleteWord(word: DictionaryWord) {
        AlertDialog.Builder(requireContext())
            .setTitle("Delete Word")
            .setMessage("Delete '${word.word}'?")
            .setPositiveButton("Delete") { _, _ ->
                lifecycleScope.launch {
                    try {
                        dataSource.deleteWord(word.word)
                        loadWords()
                    } catch (e: Exception) {
                        AlertDialog.Builder(requireContext())
                            .setTitle("Error")
                            .setMessage("Failed to delete word: ${e.message}")
                            .setPositiveButton("OK", null)
                            .show()
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showAddDialog() {
        val input = EditText(requireContext())
        input.inputType = InputType.TYPE_CLASS_TEXT
        input.hint = "Enter word"

        AlertDialog.Builder(requireContext())
            .setTitle("Add Custom Word")
            .setView(input)
            .setPositiveButton("Add") { _, _ ->
                val word = input.text.toString().trim()
                if (word.isNotBlank()) {
                    lifecycleScope.launch {
                        try {
                            dataSource.addWord(word, 100)
                            loadWords()
                        } catch (e: Exception) {
                            AlertDialog.Builder(requireContext())
                                .setTitle("Error")
                                .setMessage("Failed to add word: ${e.message}")
                                .setPositiveButton("OK", null)
                                .show()
                        }
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showEditDialog(word: DictionaryWord) {
        val input = EditText(requireContext())
        input.inputType = InputType.TYPE_CLASS_TEXT
        input.setText(word.word)
        input.selectAll()

        AlertDialog.Builder(requireContext())
            .setTitle("Edit Word")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val newWord = input.text.toString().trim()
                if (newWord.isNotBlank() && newWord != word.word) {
                    lifecycleScope.launch {
                        try {
                            dataSource.updateWord(word.word, newWord, word.frequency)
                            loadWords()
                        } catch (e: Exception) {
                            AlertDialog.Builder(requireContext())
                                .setTitle("Error")
                                .setMessage("Failed to update word: ${e.message}")
                                .setPositiveButton("OK", null)
                                .show()
                        }
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    fun refresh() {
        loadWords()
    }
}
