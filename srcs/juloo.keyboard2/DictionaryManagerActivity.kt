package juloo.keyboard2

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.MenuItem
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.Spinner
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.button.MaterialButton
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator

/**
 * Dictionary Manager Activity
 * Provides UI for managing dictionary words across multiple sources
 */
class DictionaryManagerActivity : AppCompatActivity() {

    private lateinit var toolbar: Toolbar
    private lateinit var searchInput: EditText
    private lateinit var filterSpinner: Spinner
    private lateinit var resetButton: MaterialButton
    private lateinit var tabLayout: TabLayout
    private lateinit var viewPager: ViewPager2

    private lateinit var fragments: List<WordListFragment>
    private val searchHandler = Handler(Looper.getMainLooper())
    private var searchRunnable: Runnable? = null
    private var currentSearchQuery = ""

    companion object {
        private const val SEARCH_DEBOUNCE_MS = 300L
        private val TAB_TITLES = listOf("Active", "Disabled", "User Dict", "Custom")
    }

    enum class FilterType {
        ALL, MAIN, USER, CUSTOM
    }

    private var currentFilter: FilterType = FilterType.ALL

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dictionary_manager)

        initializeViews()
        setupToolbar()
        setupViewPager()
        setupSearch()
        setupFilter()
        setupResetButton()
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun initializeViews() {
        toolbar = findViewById(R.id.toolbar)
        searchInput = findViewById(R.id.search_input)
        filterSpinner = findViewById(R.id.filter_spinner)
        resetButton = findViewById(R.id.reset_button)
        tabLayout = findViewById(R.id.tab_layout)
        viewPager = findViewById(R.id.view_pager)
    }

    private fun setupToolbar() {
        setSupportActionBar(toolbar)
        supportActionBar?.apply {
            title = "Dictionary Manager"
            setDisplayHomeAsUpEnabled(true)
        }
    }

    private fun setupViewPager() {
        // Create fragments for each tab
        fragments = listOf(
            WordListFragment.newInstance(WordListFragment.TabType.ACTIVE),
            WordListFragment.newInstance(WordListFragment.TabType.DISABLED),
            WordListFragment.newInstance(WordListFragment.TabType.USER),
            WordListFragment.newInstance(WordListFragment.TabType.CUSTOM)
        )

        // Setup ViewPager2 adapter
        viewPager.adapter = object : FragmentStateAdapter(this) {
            override fun getItemCount() = fragments.size
            override fun createFragment(position: Int) = fragments[position]
        }

        // Connect TabLayout with ViewPager2
        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            tab.text = TAB_TITLES[position]
        }.attach()
    }

    private fun setupSearch() {
        searchInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}

            override fun afterTextChanged(s: Editable?) {
                val query = s?.toString() ?: ""

                // Cancel previous search
                searchRunnable?.let { searchHandler.removeCallbacks(it) }

                // Schedule new search with debounce
                searchRunnable = Runnable {
                    currentSearchQuery = query
                    performSearch(query)
                }.also {
                    searchHandler.postDelayed(it, SEARCH_DEBOUNCE_MS)
                }
            }
        })
    }

    private fun setupFilter() {
        val filterOptions = FilterType.values().map { it.name.lowercase().capitalize() }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, filterOptions)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        filterSpinner.adapter = adapter

        filterSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                applyFilter(FilterType.values()[position])
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupResetButton() {
        resetButton.setOnClickListener {
            resetSearch()
        }
    }

    private fun performSearch(query: String) {
        val sourceFilter = when (currentFilter) {
            FilterType.ALL -> null
            FilterType.MAIN -> WordSource.MAIN
            FilterType.USER -> WordSource.USER
            FilterType.CUSTOM -> WordSource.CUSTOM
        }

        // Apply search to all fragments with source filter
        fragments.forEach { it.filter(query, sourceFilter) }

        // Auto-switch tabs if current tab has no results
        if (query.isNotEmpty() || sourceFilter != null) {
            val currentFragment = fragments[viewPager.currentItem]
            if (currentFragment.getFilteredCount() == 0) {
                // Find first tab with results
                val tabWithResults = fragments.indexOfFirst { it.getFilteredCount() > 0 }
                if (tabWithResults >= 0 && tabWithResults != viewPager.currentItem) {
                    viewPager.currentItem = tabWithResults
                }
            }
        }
    }

    private fun applyFilter(filterType: FilterType) {
        currentFilter = filterType
        performSearch(currentSearchQuery)
    }

    private fun resetSearch() {
        searchInput.setText("")
        filterSpinner.setSelection(0)  // Reset to "All"
        currentSearchQuery = ""
        performSearch("")
    }

    /**
     * Called by fragments when words are modified to refresh other tabs
     */
    fun refreshAllTabs() {
        fragments.forEach { it.refresh() }
    }
}

// Extension function to capitalize first letter
private fun String.capitalize(): String {
    return this.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
}
