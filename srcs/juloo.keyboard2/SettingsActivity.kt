package juloo.keyboard2

import android.app.AlertDialog
import android.app.ProgressDialog
import android.content.Intent
import android.content.SharedPreferences
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.preference.Preference
import android.preference.PreferenceActivity
import android.preference.PreferenceManager
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import juloo.keyboard2.ml.SwipeMLDataStore
import juloo.keyboard2.ml.SwipeMLTrainer
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*

class SettingsActivity : PreferenceActivity(), SharedPreferences.OnSharedPreferenceChangeListener {
    companion object {
        private const val TAG = "SettingsActivity"

        // Request codes for backup/restore file picker
        private const val REQUEST_CODE_BACKUP = 1001
        private const val REQUEST_CODE_RESTORE = 1002
        private const val REQUEST_CODE_NEURAL_ENCODER = 1003
        private const val REQUEST_CODE_NEURAL_DECODER = 1004
        private const val REQUEST_CODE_INSTALL_APK = 1005
        private const val REQUEST_CODE_EXPORT_CUSTOM_DICT = 1006
        private const val REQUEST_CODE_IMPORT_CUSTOM_DICT = 1007
        private const val REQUEST_CODE_EXPORT_CLIPBOARD = 1008
        private const val REQUEST_CODE_IMPORT_CLIPBOARD = 1009
    }

    private lateinit var backupRestoreManager: BackupRestoreManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // The preferences can't be read when in direct-boot mode
        try {
            val prefs = preferenceManager.sharedPreferences
            // Repair corrupted float preferences before loading preference UI
            Config.repairCorruptedFloatPreferences(prefs)
            Config.migrate(prefs)
        } catch (e: Exception) {
            fallbackEncrypted()
            return
        }

        addPreferencesFromResource(R.xml.settings)

        // Initialize backup/restore manager
        backupRestoreManager = BackupRestoreManager(this)

        // Setup handlers
        setupBackupRestoreHandlers()
        setupCGRResetButtons()
        updateCGRParameterSummaries()

        // Setup foldable device preferences
        val foldableDevice = FoldStateTracker.isFoldableDevice(this)
        findPreference("margin_bottom_portrait_unfolded").isEnabled = foldableDevice
        findPreference("margin_bottom_landscape_unfolded").isEnabled = foldableDevice
        findPreference("horizontal_margin_portrait_unfolded").isEnabled = foldableDevice
        findPreference("horizontal_margin_landscape_unfolded").isEnabled = foldableDevice
        findPreference("keyboard_height_unfolded").isEnabled = foldableDevice
        findPreference("keyboard_height_landscape_unfolded").isEnabled = foldableDevice

        // Setup version info
        setupVersionInfo()

        // Setup all preference click handlers
        setupPreferenceHandlers()

        // Update displays
        updateClipboardStats()
        updateNeuralModelInfo()
    }

    private fun setupVersionInfo() {
        val versionPref = findPreference("version_info") ?: return

        try {
            val versionInfo = loadVersionInfo()
            val commit = versionInfo.getProperty("commit", "unknown")
            val commitDate = versionInfo.getProperty("commit_date", "")
            val buildDate = versionInfo.getProperty("build_date", "")
            val buildNumber = versionInfo.getProperty("build_number", "")

            versionPref.title = "Version Info"
            versionPref.summary = String.format(
                "Build: %s\nCommit: %s (%s)\nBuilt: %s",
                buildNumber.substring(maxOf(0, buildNumber.length - 8)),
                commit, commitDate, buildDate
            )
        } catch (e: Exception) {
            versionPref.summary = "Version info unavailable"
            Log.e(TAG, "Failed to load version info", e)
        }
    }

    private fun setupPreferenceHandlers() {
        // Update app
        findPreference("update_app")?.setOnPreferenceClickListener {
            installUpdate()
            true
        }

        // Swipe calibration
        findPreference("swipe_calibration")?.setOnPreferenceClickListener {
            startActivity(Intent(this, SwipeCalibrationActivity::class.java))
            true
        }

        // Swipe debug
        findPreference("swipe_debug")?.setOnPreferenceClickListener {
            startActivity(Intent(this, SwipeDebugActivity::class.java))
            true
        }

        // Dictionary manager
        findPreference("dictionary_manager")?.setOnPreferenceClickListener {
            startActivity(Intent(this, DictionaryManagerActivity::class.java))
            true
        }

        // Neural performance statistics
        findPreference("neural_performance_stats")?.setOnPreferenceClickListener {
            showPerformanceStatistics()
            true
        }

        // Neural model metadata
        findPreference("neural_model_metadata")?.setOnPreferenceClickListener {
            showModelMetadata()
            true
        }

        // A/B test status
        findPreference("ab_test_status")?.setOnPreferenceClickListener {
            showABTestStatus()
            true
        }

        // A/B test comparison
        findPreference("ab_test_comparison")?.setOnPreferenceClickListener {
            showABTestComparison()
            true
        }

        // A/B test configuration
        findPreference("ab_test_configure")?.setOnPreferenceClickListener {
            showABTestConfiguration()
            true
        }

        // A/B test export
        findPreference("ab_test_export")?.setOnPreferenceClickListener {
            exportABTestData()
            true
        }

        // A/B test reset
        findPreference("ab_test_reset")?.setOnPreferenceClickListener {
            resetABTest()
            true
        }

        // Rollback status
        findPreference("rollback_status")?.setOnPreferenceClickListener {
            showRollbackStatus()
            true
        }

        // Rollback history
        findPreference("rollback_history")?.setOnPreferenceClickListener {
            showRollbackHistory()
            true
        }

        // Auto-rollback enabled
        findPreference("rollback_auto_enabled")?.setOnPreferenceChangeListener { _, newValue ->
            val enabled = newValue as Boolean
            ModelVersionManager.getInstance(this).setAutoRollbackEnabled(enabled)
            android.widget.Toast.makeText(
                this,
                "Auto-rollback ${if (enabled) "enabled" else "disabled"}",
                android.widget.Toast.LENGTH_SHORT
            ).show()
            true
        }

        // Manual rollback
        findPreference("rollback_manual")?.setOnPreferenceClickListener {
            performManualRollback()
            true
        }

        // Pin version
        findPreference("rollback_pin_version")?.setOnPreferenceClickListener {
            toggleVersionPin()
            true
        }

        // Export rollback history
        findPreference("rollback_export")?.setOnPreferenceClickListener {
            exportRollbackHistory()
            true
        }

        // Reset rollback data
        findPreference("rollback_reset")?.setOnPreferenceClickListener {
            resetRollbackData()
            true
        }

        // ML data export
        findPreference("export_swipe_ml_data")?.let { pref ->
            try {
                val dataStore = SwipeMLDataStore.getInstance(this)
                val stats = dataStore.getStatistics()
                pref.summary = "Export all collected swipe data (${stats.totalCount} samples)"
            } catch (e: Exception) {
                pref.summary = "Export all collected swipe data"
                Log.e(TAG, "Failed to get ML data statistics", e)
            }

            pref.setOnPreferenceClickListener {
                exportSwipeMLData()
                true
            }
        }

        // ML data import
        findPreference("import_swipe_ml_data")?.setOnPreferenceClickListener {
            importSwipeMLData()
            true
        }

        // ML training
        findPreference("train_swipe_ml_model")?.let { pref ->
            try {
                val dataStore = SwipeMLDataStore.getInstance(this)
                val stats = dataStore.getStatistics()
                pref.summary = "Train model with ${stats.totalCount} samples (min 100 required)"
            } catch (e: Exception) {
                pref.summary = "Train swipe prediction model"
                Log.e(TAG, "Failed to get ML data statistics", e)
            }

            pref.setOnPreferenceClickListener {
                startMLTraining()
                true
            }
        }

        // Model version change listener
        findPreference("neural_model_version")?.setOnPreferenceChangeListener { _, _ ->
            Handler(Looper.getMainLooper()).postDelayed({ updateNeuralModelInfo() }, 100)
            true
        }

        // Neural model file pickers
        findPreference("neural_load_encoder")?.let { pref ->
            pref.setOnPreferenceClickListener {
                openFilePicker(REQUEST_CODE_NEURAL_ENCODER)
                true
            }
            updateModelFileSummary(pref, "neural_custom_encoder_uri")
        }

        findPreference("neural_load_decoder")?.let { pref ->
            pref.setOnPreferenceClickListener {
                openFilePicker(REQUEST_CODE_NEURAL_DECODER)
                true
            }
            updateModelFileSummary(pref, "neural_custom_decoder_uri")
        }
    }

    private fun updateNeuralModelInfo() {
        val modelInfoPref = findPreference("neural_model_info") ?: return

        try {
            val prefs = preferenceManager.sharedPreferences
            val modelVersion = prefs.getString("neural_model_version", "v2")
            val encoderUri = prefs.getString("neural_custom_encoder_uri", null)
            val decoderUri = prefs.getString("neural_custom_decoder_uri", null)

            val summary = StringBuilder()

            if (modelVersion == "custom") {
                when {
                    encoderUri != null && decoderUri != null ->
                        summary.append("✅ Custom models selected\nWill load on next swipe (may take ~5s first time)")
                    encoderUri != null ->
                        summary.append("⚠️ Only encoder selected\nNeed decoder file too")
                    decoderUri != null ->
                        summary.append("⚠️ Only decoder selected\nNeed encoder file too")
                    else ->
                        summary.append("⚠️ No custom files selected\nUse file pickers below")
                }
            } else {
                summary.append("✅ Built-in model (v2)\n250-length, 80.6% accuracy, ready to use")
            }

            modelInfoPref.summary = summary.toString()
        } catch (e: Exception) {
            modelInfoPref.summary = "❌ Error loading model info"
            Log.e(TAG, "Failed to get model info", e)
        }
    }

    private fun exportSwipeMLData() {
        try {
            val dataStore = SwipeMLDataStore.getInstance(this)
            val stats = dataStore.getStatistics()

            if (stats.totalCount == 0) {
                Toast.makeText(this, "No swipe data to export", Toast.LENGTH_SHORT).show()
                return
            }

            val exportFile = dataStore.exportToJSON()

            val message = "Exported ${stats.totalCount} swipe samples\n\n" +
                    "File saved to:\n${exportFile.absolutePath}\n\n" +
                    "Statistics:\n" +
                    "• Calibration samples: ${stats.calibrationCount}\n" +
                    "• User samples: ${stats.userSelectionCount}\n" +
                    "• Unique words: ${stats.uniqueWords}"

            AlertDialog.Builder(this)
                .setTitle("Export Successful")
                .setMessage(message)
                .setPositiveButton("OK", null)
                .setNeutralButton("Copy Path") { _, _ ->
                    val clipboard = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
                    val clip = android.content.ClipData.newPlainText("Export Path", exportFile.absolutePath)
                    clipboard.setPrimaryClip(clip)
                    Toast.makeText(this, "Path copied to clipboard", Toast.LENGTH_SHORT).show()
                }
                .show()
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to export ML data", e)
        }
    }

    private fun importSwipeMLData() {
        val possibleDirs = arrayOf(
            File("/sdcard/Android/data/juloo.keyboard2.debug/files/swipe_ml_export/"),
            File("/storage/emulated/0/Android/data/juloo.keyboard2.debug/files/swipe_ml_export/"),
            File("/sdcard/Download/"),
            File("/storage/emulated/0/Download/")
        )

        val jsonFiles = possibleDirs.flatMap { dir ->
            if (dir.exists() && dir.isDirectory) {
                dir.listFiles { _, name -> name.endsWith(".json") }?.toList() ?: emptyList()
            } else emptyList()
        }

        if (jsonFiles.isEmpty()) {
            Toast.makeText(this, "No JSON files found in common locations", Toast.LENGTH_LONG).show()
            return
        }

        val fileNames = jsonFiles.map { "${it.name}\n(${it.parent})" }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Import Swipe Data")
            .setMessage("Select JSON file to import from:")
            .setItems(fileNames) { _, which -> performImport(jsonFiles[which]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun performImport(jsonFile: File) {
        try {
            val dataStore = SwipeMLDataStore.getInstance(this)
            val importedCount = dataStore.importFromJSON(jsonFile)

            if (importedCount > 0) {
                Toast.makeText(this, "Successfully imported $importedCount swipe samples",
                    Toast.LENGTH_LONG).show()

                findPreference("export_swipe_ml_data")?.let { pref ->
                    val stats = dataStore.getStatistics()
                    pref.summary = "Export all collected swipe data (${stats.totalCount} samples)"
                }
            } else {
                Toast.makeText(this, "No new samples imported (duplicates skipped)",
                    Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to import ML data", e)
        }
    }

    private fun startMLTraining() {
        val trainer = SwipeMLTrainer(this)

        if (!trainer.canTrain()) {
            Toast.makeText(this, "Not enough data for training. Need at least 100 samples.",
                Toast.LENGTH_LONG).show()
            return
        }

        val progressDialog = ProgressDialog(this).apply {
            setTitle("Training ML Model")
            setMessage("Preparing training data...")
            setProgressStyle(ProgressDialog.STYLE_HORIZONTAL)
            max = 100
            setCancelable(false)
            show()
        }

        trainer.setTrainingListener(object : SwipeMLTrainer.TrainingListener {
            override fun onTrainingStarted() {
                runOnUiThread {
                    progressDialog.setMessage("Training in progress...")
                }
            }

            override fun onTrainingProgress(progress: Int, total: Int) {
                runOnUiThread {
                    progressDialog.progress = progress
                }
            }

            override fun onTrainingCompleted(result: SwipeMLTrainer.TrainingResult) {
                runOnUiThread {
                    progressDialog.dismiss()
                    val message = String.format(
                        "Training completed!\nSamples: %d\nTime: %.1f seconds\nAccuracy: %.1f%%",
                        result.samplesUsed, result.trainingTimeMs / 1000.0, result.accuracy * 100
                    )
                    Toast.makeText(this@SettingsActivity, message, Toast.LENGTH_LONG).show()
                }
            }

            override fun onTrainingError(error: String) {
                runOnUiThread {
                    progressDialog.dismiss()
                    Toast.makeText(this@SettingsActivity, "Training failed: $error",
                        Toast.LENGTH_LONG).show()
                }
            }
        })

        trainer.startTraining()
    }

    private fun loadVersionInfo(): Properties {
        val props = Properties()
        try {
            val reader = BufferedReader(
                InputStreamReader(resources.openRawResource(
                    resources.getIdentifier("version_info", "raw", packageName)
                ))
            )
            props.load(reader)
            reader.close()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load version info", e)
        }
        return props
    }

    private fun installUpdate() {
        val unexpectedDir = File("/storage/emulated/0/unexpected")

        if (!unexpectedDir.exists() || !unexpectedDir.isDirectory) {
            Toast.makeText(this, "📂 Directory not found. Opening file picker...",
                Toast.LENGTH_SHORT).show()
            openApkFilePicker()
            return
        }

        val apkFiles = unexpectedDir.listFiles { _, name ->
            name.lowercase().endsWith(".apk")
        }

        if (apkFiles == null || apkFiles.isEmpty()) {
            Toast.makeText(this, "📂 No APKs in /storage/emulated/0/unexpected/. Opening file picker...",
                Toast.LENGTH_SHORT).show()
            openApkFilePicker()
            return
        }

        // Sort by modification time (newest first)
        apkFiles.sortByDescending { it.lastModified() }

        val apkNames = apkFiles.map { apk ->
            val sizeMB = apk.length() / (1024 * 1024)
            val date = SimpleDateFormat("MM-dd HH:mm", Locale.US).format(Date(apk.lastModified()))
            String.format("%s\n%d MB • %s", apk.name, sizeMB, date)
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("📦 Select APK to Install")
            .setItems(apkNames) { _, which -> installApkFile(apkFiles[which]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun installApkFile(apkFile: File) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                if (!packageManager.canRequestPackageInstalls()) {
                    val intent = Intent(android.provider.Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES)
                    intent.data = Uri.parse("package:$packageName")
                    Toast.makeText(this, "⚠️ Please allow installing from this source",
                        Toast.LENGTH_LONG).show()
                    startActivity(intent)
                    return
                }
            }

            val intent = Intent(Intent.ACTION_VIEW)
            val apkUri = Uri.fromFile(apkFile)
            intent.setDataAndType(apkUri, "application/vnd.android.package-archive")
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION

            Log.d(TAG, "Installing APK: ${apkFile.absolutePath}")
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to install APK", e)
            showInstallAlternatives(apkFile, e)
        }
    }

    private fun installApkFromUri(apkUri: Uri, filename: String?) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                if (!packageManager.canRequestPackageInstalls()) {
                    val intent = Intent(android.provider.Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES)
                    intent.data = Uri.parse("package:$packageName")
                    Toast.makeText(this, "⚠️ Please allow installing from this source",
                        Toast.LENGTH_LONG).show()
                    startActivity(intent)
                    return
                }
            }

            val intent = Intent(Intent.ACTION_VIEW)
            intent.setDataAndType(apkUri, "application/vnd.android.package-archive")
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION

            Log.d(TAG, "Installing APK from URI: $filename")
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to install APK from URI", e)
            Toast.makeText(this, "❌ Install failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun showInstallAlternatives(apkFile: File, error: Exception) {
        AlertDialog.Builder(this)
            .setTitle("⚠️ Installation Failed")
            .setMessage("Could not open installer:\n${error.message}\n\nChoose an alternative:")
            .setPositiveButton("📂 Open with File Manager") { _, _ ->
                try {
                    val intent = Intent(Intent.ACTION_VIEW)
                    intent.setDataAndType(Uri.fromFile(apkFile.parentFile), "resource/folder")
                    startActivity(intent)
                } catch (e: Exception) {
                    Toast.makeText(this, "Could not open file manager: ${e.message}",
                        Toast.LENGTH_LONG).show()
                }
            }
            .setNeutralButton("📋 Copy Path") { _, _ ->
                val clipboard = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
                val clip = android.content.ClipData.newPlainText("APK Path", apkFile.absolutePath)
                clipboard.setPrimaryClip(clip)
                Toast.makeText(this, "✅ Path copied: ${apkFile.absolutePath}",
                    Toast.LENGTH_LONG).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun fallbackEncrypted() {
        finish()
    }

    override fun onStop() {
        DirectBootAwarePreferences.copy_preferences_to_protected_storage(
            this, preferenceManager.sharedPreferences
        )
        super.onStop()
    }

    private fun setupBackupRestoreHandlers() {
        findPreference("backup_config")?.setOnPreferenceClickListener {
            startBackup()
            true
        }

        findPreference("restore_config")?.setOnPreferenceClickListener {
            startRestore()
            true
        }

        findPreference("export_custom_dictionary")?.setOnPreferenceClickListener {
            startExportCustomDictionary()
            true
        }

        findPreference("import_custom_dictionary")?.setOnPreferenceClickListener {
            startImportCustomDictionary()
            true
        }

        findPreference("export_clipboard_history")?.setOnPreferenceClickListener {
            startExportClipboardHistory()
            true
        }

        findPreference("import_clipboard_history")?.setOnPreferenceClickListener {
            startImportClipboardHistory()
            true
        }
    }

    private fun startBackup() {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "kb-config-$timestamp.json"

        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/json"
            putExtra(Intent.EXTRA_TITLE, fileName)
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_BACKUP)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start backup", e)
        }
    }

    private fun startRestore() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/json"
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_RESTORE)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start restore", e)
        }
    }

    private fun openFilePicker(requestCode: Int) {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }

        try {
            startActivityForResult(intent, requestCode)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start file picker", e)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode != RESULT_OK || data == null) return

        val uri = data.data ?: return

        when (requestCode) {
            REQUEST_CODE_BACKUP -> performBackup(uri)
            REQUEST_CODE_RESTORE -> performRestore(uri)
            REQUEST_CODE_EXPORT_CUSTOM_DICT -> performExportCustomDictionary(uri)
            REQUEST_CODE_IMPORT_CUSTOM_DICT -> performImportCustomDictionary(uri)
            REQUEST_CODE_EXPORT_CLIPBOARD -> performExportClipboardHistory(uri)
            REQUEST_CODE_IMPORT_CLIPBOARD -> performImportClipboardHistory(uri)
            REQUEST_CODE_NEURAL_ENCODER -> handleNeuralModelFile(uri, true)
            REQUEST_CODE_NEURAL_DECODER -> handleNeuralModelFile(uri, false)
            REQUEST_CODE_INSTALL_APK -> handleApkFileSelection(uri)
        }
    }

    private fun openApkFilePicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/vnd.android.package-archive"
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_INSTALL_APK)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start APK file picker", e)
        }
    }

    private fun handleApkFileSelection(uri: Uri) {
        try {
            val filename = getFilenameFromUri(uri)

            if (filename != null && !filename.lowercase().endsWith(".apk")) {
                Toast.makeText(this, "❌ File must be an .apk file", Toast.LENGTH_SHORT).show()
                return
            }

            installApkFromUri(uri, filename)
        } catch (e: Exception) {
            Toast.makeText(this, "Error selecting APK: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Error handling APK file selection", e)
        }
    }

    private fun handleNeuralModelFile(uri: Uri, isEncoder: Boolean) {
        try {
            // Take persistent permission
            val takeFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION
            contentResolver.takePersistableUriPermission(uri, takeFlags)

            // Validate file access
            var fileSize: Long = 0
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val header = ByteArray(4)
                val bytesRead = inputStream.read(header)
                if (bytesRead < 4) {
                    Toast.makeText(this, "File is empty or unreadable", Toast.LENGTH_LONG).show()
                    return
                }
                fileSize = inputStream.available().toLong() + bytesRead
            } ?: run {
                Toast.makeText(this, "Cannot access file", Toast.LENGTH_LONG).show()
                return
            }

            val filename = getFilenameFromUri(uri)
            val fileSizeAccurate = getFileSizeFromUri(uri)
            if (fileSizeAccurate > 0) fileSize = fileSizeAccurate

            if (filename != null && !filename.lowercase().endsWith(".onnx")) {
                Toast.makeText(this, "File must be an .onnx file", Toast.LENGTH_SHORT).show()
                return
            }

            // Save URI to preferences
            val editor = preferenceManager.sharedPreferences.edit()
            val prefKey = if (isEncoder) "neural_custom_encoder_uri" else "neural_custom_decoder_uri"
            editor.putString(prefKey, uri.toString())
            editor.apply()

            val sizeStr = formatFileSize(fileSize)

            // Update preference summary
            val prefItemKey = if (isEncoder) "neural_load_encoder" else "neural_load_decoder"
            findPreference(prefItemKey)?.summary = "✅ $filename ($sizeStr)"

            Toast.makeText(this, "✅ ${if (isEncoder) "Encoder" else "Decoder"} file selected: $filename ($sizeStr)",
                Toast.LENGTH_LONG).show()

            // Check if both files are set
            val prefs = preferenceManager.sharedPreferences
            val encoderUri = prefs.getString("neural_custom_encoder_uri", null)
            val decoderUri = prefs.getString("neural_custom_decoder_uri", null)
            val modelVersion = prefs.getString("neural_model_version", "v2")

            if (encoderUri != null && decoderUri != null) {
                if (modelVersion != "custom") {
                    Toast.makeText(this, "📝 Both files loaded! Change 'Model Version' to 'custom' to use them.",
                        Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this, "🔄 Custom models will load on next swipe. This may take a moment...",
                        Toast.LENGTH_LONG).show()
                }
            }

            updateNeuralModelInfo()
        } catch (e: SecurityException) {
            Toast.makeText(this, "❌ Permission denied. Please grant access to the file.",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Security exception loading model", e)
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Error loading neural model file", e)
        }
    }

    private fun performBackup(uri: Uri) {
        try {
            val prefs = preferenceManager.sharedPreferences
            val success = backupRestoreManager.exportConfig(uri, prefs)

            if (success) {
                val count = prefs.all.size
                Toast.makeText(this, "Successfully exported $count settings",
                    Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Backup failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Backup failed", e)
        }
    }

    private fun performRestore(uri: Uri) {
        try {
            val prefs = preferenceManager.sharedPreferences
            val result = backupRestoreManager.importConfig(uri, prefs)
            showRestoreResults(result)
        } catch (e: Exception) {
            Toast.makeText(this, "Restore failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Restore failed", e)
        }
    }

    // Due to token constraints, I need to continue in the next part
    // The remaining methods follow the same pattern - converting Java to Kotlin idiomatically

    private fun startExportCustomDictionary() {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "custom-dictionary-$timestamp.json"

        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/json"
            putExtra(Intent.EXTRA_TITLE, fileName)
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_EXPORT_CUSTOM_DICT)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start custom dictionary export", e)
        }
    }

    private fun performExportCustomDictionary(uri: Uri) {
        try {
            val prefs = DirectBootAwarePreferences.get_shared_preferences(this)

            val customWordsJson = prefs.getString("custom_words", "{}")
            val customWords = JSONObject(customWordsJson!!)
            val customWordCount = customWords.length()

            val disabledWordsSet = prefs.getStringSet("disabled_words", HashSet())
            val disabledWords = JSONArray(disabledWordsSet)
            val disabledWordCount = disabledWordsSet!!.size

            if (customWordCount == 0 && disabledWordCount == 0) {
                Toast.makeText(this, "No custom or disabled words to export",
                    Toast.LENGTH_SHORT).show()
                return
            }

            val exportData = JSONObject().apply {
                put("custom_words", customWords)
                put("disabled_words", disabledWords)
                put("export_version", 1)
                put("export_date", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date()))
            }

            contentResolver.openOutputStream(uri)?.use { outputStream ->
                val prettyJson = exportData.toString(2)
                outputStream.write(prettyJson.toByteArray())
            } ?: throw java.io.IOException("Failed to open output stream")

            val message = "Successfully exported:\n" +
                    "• $customWordCount custom word${if (customWordCount == 1) "" else "s"}\n" +
                    "• $disabledWordCount disabled word${if (disabledWordCount == 1) "" else "s"}"
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            Log.d(TAG, "Exported $customWordCount custom words and $disabledWordCount disabled words")
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Custom dictionary export failed", e)
        }
    }

    private fun startImportCustomDictionary() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/json"
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_IMPORT_CUSTOM_DICT)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start custom dictionary import", e)
        }
    }

    private fun performImportCustomDictionary(uri: Uri) {
        try {
            val jsonContent = StringBuilder()
            contentResolver.openInputStream(uri)?.use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).forEachLine { line ->
                    jsonContent.append(line)
                }
            } ?: throw java.io.IOException("Failed to open input stream")

            val importData = JSONObject(jsonContent.toString())

            val prefs = DirectBootAwarePreferences.get_shared_preferences(this)
            val editor = prefs.edit()

            var customWordsAdded = 0
            var customWordsUpdated = 0
            if (importData.has("custom_words")) {
                val importCustomWords = importData.getJSONObject("custom_words")

                val existingJson = prefs.getString("custom_words", "{}")
                val existingCustomWords = JSONObject(existingJson!!)

                val keys = importCustomWords.keys()
                while (keys.hasNext()) {
                    val word = keys.next()
                    val importedFreq = importCustomWords.getInt(word)

                    if (existingCustomWords.has(word)) {
                        val existingFreq = existingCustomWords.getInt(word)
                        if (importedFreq > existingFreq) {
                            existingCustomWords.put(word, importedFreq)
                            customWordsUpdated++
                        }
                    } else {
                        existingCustomWords.put(word, importedFreq)
                        customWordsAdded++
                    }
                }

                editor.putString("custom_words", existingCustomWords.toString())
            }

            var disabledWordsAdded = 0
            if (importData.has("disabled_words")) {
                val importDisabledWords = importData.getJSONArray("disabled_words")

                val existingDisabled = HashSet(prefs.getStringSet("disabled_words", HashSet()))
                val initialSize = existingDisabled.size

                for (i in 0 until importDisabledWords.length()) {
                    existingDisabled.add(importDisabledWords.getString(i))
                }

                disabledWordsAdded = existingDisabled.size - initialSize
                editor.putStringSet("disabled_words", existingDisabled)
            }

            editor.apply()

            val message = buildString {
                append("Import complete:\n")
                if (customWordsAdded > 0 || customWordsUpdated > 0) {
                    append("• Custom words: $customWordsAdded added")
                    if (customWordsUpdated > 0) {
                        append(", $customWordsUpdated updated")
                    }
                    append("\n")
                }
                if (disabledWordsAdded > 0) {
                    append("• Disabled words: $disabledWordsAdded added\n")
                }
                if (customWordsAdded == 0 && customWordsUpdated == 0 && disabledWordsAdded == 0) {
                    append("• No new words (all already exist)")
                }
            }

            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            Log.d(TAG, "Imported: $customWordsAdded custom words added, $customWordsUpdated updated, $disabledWordsAdded disabled words added")
        } catch (e: Exception) {
            Toast.makeText(this, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Custom dictionary import failed", e)
        }
    }

    private fun startExportClipboardHistory() {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "clipboard-history-$timestamp.json"

        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/json"
            putExtra(Intent.EXTRA_TITLE, fileName)
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_EXPORT_CLIPBOARD)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start clipboard export", e)
        }
    }

    private fun performExportClipboardHistory(uri: Uri) {
        try {
            val db = ClipboardDatabase.getInstance(this)
            val exportData = db.exportToJSON() ?: run {
                Toast.makeText(this, "Failed to export clipboard history", Toast.LENGTH_SHORT).show()
                return
            }

            val activeCount = exportData.getInt("total_active")
            val pinnedCount = exportData.getInt("total_pinned")

            if (activeCount == 0 && pinnedCount == 0) {
                Toast.makeText(this, "No clipboard entries to export", Toast.LENGTH_SHORT).show()
                return
            }

            contentResolver.openOutputStream(uri)?.use { outputStream ->
                val prettyJson = exportData.toString(2)
                outputStream.write(prettyJson.toByteArray())
            } ?: throw java.io.IOException("Failed to open output stream")

            val message = "Successfully exported:\n" +
                    "• $activeCount active entr${if (activeCount == 1) "y" else "ies"}\n" +
                    "• $pinnedCount pinned entr${if (pinnedCount == 1) "y" else "ies"}"
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            Log.d(TAG, "Exported $activeCount active and $pinnedCount pinned clipboard entries")
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Clipboard export failed", e)
        }
    }

    private fun startImportClipboardHistory() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/json"
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_IMPORT_CLIPBOARD)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to open file picker: ${e.message}",
                Toast.LENGTH_LONG).show()
            Log.e(TAG, "Failed to start clipboard import", e)
        }
    }

    private fun performImportClipboardHistory(uri: Uri) {
        try {
            val jsonContent = StringBuilder()
            contentResolver.openInputStream(uri)?.use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).forEachLine { line ->
                    jsonContent.append(line)
                }
            } ?: throw java.io.IOException("Failed to open input stream")

            val importData = JSONObject(jsonContent.toString())

            val db = ClipboardDatabase.getInstance(this)
            val results = db.importFromJSON(importData)

            val activeAdded = results[0]
            val pinnedAdded = results[1]
            val duplicatesSkipped = results[2]

            val message = buildString {
                append("Import complete:\n")
                if (activeAdded > 0) {
                    append("• $activeAdded active entr${if (activeAdded == 1) "y" else "ies"} added\n")
                }
                if (pinnedAdded > 0) {
                    append("• $pinnedAdded pinned entr${if (pinnedAdded == 1) "y" else "ies"} added\n")
                }
                if (duplicatesSkipped > 0) {
                    append("• $duplicatesSkipped duplicate${if (duplicatesSkipped == 1) "" else "s"} skipped\n")
                }
                if (activeAdded == 0 && pinnedAdded == 0) {
                    append("• No new entries (all already exist)")
                }
            }

            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            Log.d(TAG, "Imported: $activeAdded active, $pinnedAdded pinned, $duplicatesSkipped duplicates skipped")
        } catch (e: Exception) {
            Toast.makeText(this, "Import failed: ${e.message}", Toast.LENGTH_LONG).show()
            Log.e(TAG, "Clipboard import failed", e)
        }
    }

    private fun showRestoreResults(result: BackupRestoreManager.ImportResult) {
        val message = buildString {
            append("Successfully restored ${result.importedCount} settings")

            if (result.skippedCount > 0) {
                append("\n\nSkipped ${result.skippedCount} invalid or unrecognized settings")
            }

            val skippedLayouts = result.skippedKeys.contains("layouts")
            val skippedExtraKeys = result.skippedKeys.contains("extra_keys") ||
                    result.skippedKeys.contains("custom_extra_keys")

            if (skippedLayouts || skippedExtraKeys) {
                append("\n\n⚠️ Not restored:")
                if (skippedLayouts) {
                    append("\n  • Keyboard layouts (needs manual reconfiguration)")
                }
                if (skippedExtraKeys) {
                    append("\n  • Custom extra keys (needs manual reconfiguration)")
                }
            }

            if (result.sourceVersion != "unknown") {
                append("\n\nSource version: ${result.sourceVersion}")
            }

            if (result.hasScreenSizeMismatch()) {
                append("\n\n⚠️ Warning: Backup was from a device with different screen size. " +
                        "Layout settings may need adjustment.")
            }

            append("\n\nThe app needs to restart to apply all settings correctly.")
        }

        AlertDialog.Builder(this)
            .setTitle("Restore Complete")
            .setMessage(message)
            .setCancelable(false)
            .setPositiveButton("Restart Now") { _, _ ->
                val intent = baseContext.packageManager.getLaunchIntentForPackage(baseContext.packageName)
                intent?.let {
                    it.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK)
                    startActivity(it)
                }
                finish()
                System.exit(0)
            }
            .setNegativeButton("Later") { _, _ ->
                Toast.makeText(this, "Restart the app to apply all settings",
                    Toast.LENGTH_LONG).show()
            }
            .show()
    }

    private fun setupCGRResetButtons() {
        findPreference("swipe_reset_optimal")?.setOnPreferenceClickListener {
            resetToSwipeOptimal()
            true
        }

        findPreference("swipe_reset_strict")?.setOnPreferenceClickListener {
            resetToSwipeStrict()
            true
        }

        findPreference("reset_swipe_corrections")?.setOnPreferenceClickListener {
            resetSwipeCorrections()
            true
        }
    }

    private fun resetToSwipeOptimal() {
        val prefs = preferenceManager.sharedPreferences
        prefs.edit().apply {
            putInt("proximity_weight", 100)
            putInt("missing_key_penalty", 1000)
            putInt("extra_key_penalty", 200)
            putInt("order_penalty", 500)
            putInt("start_point_weight", 300)
            putInt("key_zone_radius", 120)
            putInt("path_sample_distance", 10)
            apply()
        }

        recreate()
        Toast.makeText(this, "Reset to optimal keyboard recognition values",
            Toast.LENGTH_SHORT).show()
    }

    private fun resetToSwipeStrict() {
        val prefs = preferenceManager.sharedPreferences
        prefs.edit().apply {
            putInt("proximity_weight", 200)
            putInt("missing_key_penalty", 1500)
            putInt("extra_key_penalty", 500)
            putInt("order_penalty", 1000)
            putInt("start_point_weight", 500)
            putInt("key_zone_radius", 80)
            putInt("path_sample_distance", 5)
            apply()
        }

        recreate()
        Toast.makeText(this, "Reset to strict recognition values", Toast.LENGTH_SHORT).show()
    }

    private fun resetSwipeCorrections() {
        val prefs = preferenceManager.sharedPreferences
        prefs.edit().apply {
            putString("swipe_correction_preset", "balanced")
            putInt("autocorrect_max_length_diff", 2)
            putInt("autocorrect_prefix_length", 2)
            putInt("autocorrect_max_beam_candidates", 3)
            putFloat("autocorrect_char_match_threshold", 0.67f)
            putInt("autocorrect_confidence_min_frequency", 500)
            putBoolean("swipe_beam_autocorrect_enabled", true)
            putBoolean("swipe_final_autocorrect_enabled", true)
            putString("swipe_fuzzy_match_mode", "edit_distance")
            putInt("swipe_prediction_source", 60)
            putFloat("swipe_common_words_boost", 1.3f)
            putFloat("swipe_top5000_boost", 1.0f)
            putFloat("swipe_rare_words_penalty", 0.75f)
            apply()
        }

        recreate()
        Toast.makeText(this, "Reset all swipe correction settings to defaults",
            Toast.LENGTH_LONG).show()
        Log.d(TAG, "Reset swipe corrections to default values")
    }

    override fun onSharedPreferenceChanged(prefs: SharedPreferences?, key: String?) {
        if (key == "swipe_correction_preset") {
            val preset = prefs?.getString(key, "balanced") ?: "balanced"
            applySwipeCorrectionPreset(prefs!!, preset)
            Toast.makeText(this, "Applied \"$preset\" correction preset", Toast.LENGTH_SHORT).show()
            return
        }

        if (key != null && key.startsWith("cgr_")) {
            Log.d(TAG, "CGR parameter changed: $key")
            val currentValue = prefs?.getInt(key, -1)
            Toast.makeText(this, "Updated $key = $currentValue (takes effect immediately)",
                Toast.LENGTH_LONG).show()
            updateCGRParameterSummaries()
        }
    }

    private fun applySwipeCorrectionPreset(prefs: SharedPreferences, preset: String) {
        prefs.edit().apply {
            when (preset) {
                "strict" -> {
                    putInt("autocorrect_max_length_diff", 1)
                    putInt("autocorrect_prefix_length", 3)
                    putInt("autocorrect_max_beam_candidates", 2)
                    putFloat("autocorrect_char_match_threshold", 0.80f)
                    Log.d(TAG, "Applied STRICT preset")
                }
                "lenient" -> {
                    putInt("autocorrect_max_length_diff", 4)
                    putInt("autocorrect_prefix_length", 1)
                    putInt("autocorrect_max_beam_candidates", 5)
                    putFloat("autocorrect_char_match_threshold", 0.55f)
                    Log.d(TAG, "Applied LENIENT preset")
                }
                else -> {
                    putInt("autocorrect_max_length_diff", 2)
                    putInt("autocorrect_prefix_length", 2)
                    putInt("autocorrect_max_beam_candidates", 3)
                    putFloat("autocorrect_char_match_threshold", 0.67f)
                    Log.d(TAG, "Applied BALANCED preset")
                }
            }
            apply()
        }
    }

    override fun onResume() {
        super.onResume()
        preferenceManager.sharedPreferences.registerOnSharedPreferenceChangeListener(this)
        updateCGRParameterSummaries()
        updateClipboardStats()
    }

    override fun onPause() {
        super.onPause()
        preferenceManager.sharedPreferences.unregisterOnSharedPreferenceChangeListener(this)
    }

    private fun updateCGRParameterSummaries() {
        val prefs = preferenceManager.sharedPreferences

        findPreference("cgr_e_sigma_config")?.let { pref ->
            val eSigma = prefs.getInt("cgr_e_sigma", 120)
            pref.summary = "Current: $eSigma (Position tolerance)"
        }

        findPreference("cgr_beta_config")?.let { pref ->
            val beta = prefs.getInt("cgr_beta", 400)
            pref.summary = "Current: $beta (Variance ratio)"
        }

        findPreference("cgr_lambda_config")?.let { pref ->
            val lambda = prefs.getInt("cgr_lambda", 65)
            pref.summary = "Current: $lambda% (Distance balance)"
        }

        findPreference("cgr_kappa_config")?.let { pref ->
            val kappa = prefs.getInt("cgr_kappa", 25)
            pref.summary = "Current: ${kappa / 10.0} (End-point bias)"
        }

        findPreference("cgr_length_config")?.let { pref ->
            val lengthFilter = prefs.getInt("cgr_length_filter", 70)
            pref.summary = "Current: $lengthFilter% (Length similarity filter)"
        }
    }

    private fun updateClipboardStats() {
        val statsPref = findPreference("clipboard_storage_stats") ?: return

        try {
            val service = ClipboardHistoryService.get_service(this)
            statsPref.summary = if (service != null) {
                service.getStorageStats()
            } else {
                "Clipboard service not available"
            }
        } catch (e: Exception) {
            statsPref.summary = "Error loading statistics"
            Log.e(TAG, "Failed to load clipboard stats", e)
        }
    }

    private fun updateModelFileSummary(pref: Preference, uriPrefKey: String) {
        try {
            val prefs = preferenceManager.sharedPreferences
            val uriStr = prefs.getString(uriPrefKey, null)

            if (uriStr == null) {
                pref.summary = "No file selected"
                return
            }

            val uri = Uri.parse(uriStr)
            val filename = getFilenameFromUri(uri)
            val fileSize = getFileSizeFromUri(uri)

            if (filename != null) {
                val sizeStr = formatFileSize(fileSize)
                pref.summary = "✅ $filename ($sizeStr)"
            } else {
                pref.summary = "File selected"
            }
        } catch (e: Exception) {
            pref.summary = "Error reading file info"
            Log.e(TAG, "Failed to update model file summary", e)
        }
    }

    private fun getFilenameFromUri(uri: Uri): String? {
        var filename: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) {
                    filename = cursor.getString(nameIndex)
                }
            }
        }
        return filename ?: uri.lastPathSegment
    }

    private fun getFileSizeFromUri(uri: Uri): Long {
        var fileSize: Long = 0
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (sizeIndex >= 0) {
                    fileSize = cursor.getLong(sizeIndex)
                }
            }
        }
        return fileSize
    }

    private fun formatFileSize(bytes: Long): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> String.format("%.1f KB", bytes / 1024.0)
            else -> String.format("%.1f MB", bytes / (1024.0 * 1024.0))
        }
    }

    /**
     * Show model metadata dialog
     * @since v1.32.897 - Phase 6.2: Model Versioning
     */
    private fun showModelMetadata() {
        val metadata = NeuralModelMetadata.getInstance(this)
        val message = metadata.formatSummary()

        android.app.AlertDialog.Builder(this)
            .setTitle("🔍 Model Information")
            .setMessage(message)
            .setPositiveButton("Close", null)
            .show()
    }

    /**
     * Show performance statistics dialog with reset option
     * @since v1.32.896
     */
    private fun showPerformanceStatistics() {
        val stats = NeuralPerformanceStats.getInstance(this)
        val message = if (stats.hasStats()) {
            stats.formatSummary()
        } else {
            "No statistics available yet.\n\nStart using swipe typing with neural predictions to collect performance data!"
        }

        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("📊 Neural Performance Statistics")
            .setMessage(message)
            .setPositiveButton("Close", null)

        // Add reset button if there are stats
        if (stats.hasStats()) {
            dialog.setNeutralButton("Reset Statistics") { _, _ ->
                android.app.AlertDialog.Builder(this)
                    .setTitle("Reset Statistics?")
                    .setMessage("This will permanently delete all performance statistics. Are you sure?")
                    .setPositiveButton("Reset") { _, _ ->
                        stats.reset()
                        android.widget.Toast.makeText(
                            this,
                            "Statistics reset successfully",
                            android.widget.Toast.LENGTH_SHORT
                        ).show()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
        }

        dialog.show()
    }

    private fun showABTestStatus() {
        val manager = ABTestManager.getInstance(this)
        val message = manager.formatTestStatus()

        android.app.AlertDialog.Builder(this)
            .setTitle("🧪 A/B Test Status")
            .setMessage(message)
            .setPositiveButton("Close", null)
            .show()
    }

    private fun showABTestComparison() {
        val manager = ABTestManager.getInstance(this)
        val config = manager.getTestConfig()

        if (!config.enabled || config.modelAId.isEmpty() || config.modelBId.isEmpty()) {
            android.app.AlertDialog.Builder(this)
                .setTitle("No Active Test")
                .setMessage("Configure and start an A/B test first to see comparison results.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val tracker = ModelComparisonTracker.getInstance(this)
        val message = tracker.formatComparisonSummary(config.modelAId, config.modelBId)

        android.app.AlertDialog.Builder(this)
            .setTitle("📈 Model Comparison")
            .setMessage(message)
            .setPositiveButton("Close", null)
            .setNeutralButton("Export JSON") { _, _ ->
                exportABTestData()
            }
            .show()
    }

    private fun showABTestConfiguration() {
        val manager = ABTestManager.getInstance(this)
        val config = manager.getTestConfig()

        // For now, show current configuration and provide basic controls
        val message = if (config.enabled) {
            buildString {
                append("Current A/B Test:\n\n")
                append("Model A: ${config.modelAName}\n")
                append("Model B: ${config.modelBName}\n")
                append("Traffic Split: ${config.trafficSplitA}/${100-config.trafficSplitA}\n")
                append("Duration: ${config.testDurationDays} days\n")
                append("Min Samples: ${config.minSamplesRequired}\n")
                append("Session-based: ${if (config.sessionBased) "Yes" else "No"}\n")
                append("Auto-select winner: ${if (config.autoSelectWinner) "Yes" else "No"}\n\n")
                append("Test is currently ACTIVE")
            }
        } else {
            "No A/B test currently configured.\n\n" +
            "To set up an A/B test:\n" +
            "1. Load two different models\n" +
            "2. Use ABTestManager.configureTest() programmatically\n" +
            "3. Monitor results in Test Status\n\n" +
            "Note: Full UI configuration coming in future update!"
        }

        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("⚙️ Test Configuration")
            .setMessage(message)
            .setPositiveButton("Close", null)

        if (config.enabled) {
            dialog.setNeutralButton("Stop Test") { _, _ ->
                android.app.AlertDialog.Builder(this)
                    .setTitle("Stop Test?")
                    .setMessage("This will end the current A/B test without selecting a winner. Continue?")
                    .setPositiveButton("Stop") { _, _ ->
                        manager.stopTest()
                        android.widget.Toast.makeText(
                            this,
                            "A/B test stopped",
                            android.widget.Toast.LENGTH_SHORT
                        ).show()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }

            dialog.setNegativeButton("Select Winner") { _, _ ->
                android.app.AlertDialog.Builder(this)
                    .setTitle("Select Winner?")
                    .setMessage("This will analyze the results and select the best performing model. Continue?")
                    .setPositiveButton("Analyze") { _, _ ->
                        val winner = manager.selectWinnerAndEndTest()
                        val winnerName = if (winner == config.modelAId) {
                            config.modelAName
                        } else if (winner == config.modelBId) {
                            config.modelBName
                        } else {
                            null
                        }

                        val resultMessage = if (winnerName != null) {
                            "Winner selected: $winnerName\n\nThe test has been ended."
                        } else {
                            "No clear winner could be determined.\nThe test has been ended."
                        }

                        android.app.AlertDialog.Builder(this)
                            .setTitle("Test Results")
                            .setMessage(resultMessage)
                            .setPositiveButton("OK", null)
                            .show()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
        }

        dialog.show()
    }

    private fun exportABTestData() {
        val tracker = ModelComparisonTracker.getInstance(this)
        val jsonData = tracker.exportComparisonData()

        // Copy to clipboard
        val clipboard = getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
        val clip = android.content.ClipData.newPlainText("A/B Test Data", jsonData)
        clipboard.setPrimaryClip(clip)

        android.app.AlertDialog.Builder(this)
            .setTitle("💾 Data Exported")
            .setMessage("A/B test comparison data has been copied to clipboard as JSON.\n\nYou can paste it into a file or analysis tool.")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun resetABTest() {
        android.app.AlertDialog.Builder(this)
            .setTitle("Reset A/B Test?")
            .setMessage("This will permanently delete:\n• All test configuration\n• All collected comparison data\n• Model performance metrics\n\nThis cannot be undone. Continue?")
            .setPositiveButton("Reset") { _, _ ->
                val manager = ABTestManager.getInstance(this)
                manager.resetTest()

                android.widget.Toast.makeText(
                    this,
                    "A/B test reset successfully",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showRollbackStatus() {
        val manager = ModelVersionManager.getInstance(this)
        val message = manager.formatStatus()

        android.app.AlertDialog.Builder(this)
            .setTitle("🔙 Rollback Status")
            .setMessage(message)
            .setPositiveButton("Close", null)
            .show()
    }

    private fun showRollbackHistory() {
        val manager = ModelVersionManager.getInstance(this)
        val history = manager.getVersionHistory()

        if (history.isEmpty()) {
            android.app.AlertDialog.Builder(this)
                .setTitle("Version History")
                .setMessage("No version history available yet.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val sb = StringBuilder()
        sb.append("📜 Model Version History\n\n")

        for ((index, versionId) in history.withIndex()) {
            val version = manager.getVersion(versionId) ?: continue
            val isCurrent = manager.getCurrentVersion()?.versionId == versionId
            val isPrevious = manager.getPreviousVersion()?.versionId == versionId

            sb.append("${index + 1}. ${version.versionName}\n")
            if (isCurrent) sb.append("   [CURRENT]\n")
            if (isPrevious) sb.append("   [PREVIOUS]\n")
            if (version.isPinned) sb.append("   📌 PINNED\n")
            sb.append("   Success Rate: ${String.format("%.1f", version.getSuccessRate() * 100)}%\n")
            sb.append("   Attempts: ${version.successCount + version.failureCount}\n")
            sb.append("   Status: ${if (version.isHealthy()) "✅ Healthy" else "⚠️ Unhealthy"}\n")
            sb.append("\n")
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("📜 Version History")
            .setMessage(sb.toString())
            .setPositiveButton("Close", null)
            .setNeutralButton("Export JSON") { _, _ ->
                exportRollbackHistory()
            }
            .show()
    }

    private fun performManualRollback() {
        val manager = ModelVersionManager.getInstance(this)
        val decision = manager.shouldRollback()
        val previous = manager.getPreviousVersion()

        if (previous == null) {
            android.app.AlertDialog.Builder(this)
                .setTitle("Rollback Not Available")
                .setMessage("No previous version available to rollback to.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val message = buildString {
            append("⚠️ Manual Rollback\n\n")
            append("This will switch from the current version to:\n")
            append("${previous.versionName}\n\n")
            append("Current version health:\n")
            val current = manager.getCurrentVersion()
            if (current != null) {
                append("Success rate: ${String.format("%.1f", current.getSuccessRate() * 100)}%\n")
                append("Failures: ${current.failureCount}\n\n")
            }
            append("Note: The keyboard will need to be restarted for changes to take effect.\n\n")
            append("Continue?")
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("⚠️ Manual Rollback")
            .setMessage(message)
            .setPositiveButton("Rollback") { _, _ ->
                if (manager.rollback()) {
                    android.app.AlertDialog.Builder(this)
                        .setTitle("Rollback Complete")
                        .setMessage("Rolled back to: ${previous.versionName}\n\nPlease restart the keyboard for changes to take effect.")
                        .setPositiveButton("OK", null)
                        .show()
                } else {
                    android.app.AlertDialog.Builder(this)
                        .setTitle("Rollback Failed")
                        .setMessage("Could not perform rollback. Check logs for details.")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun toggleVersionPin() {
        val manager = ModelVersionManager.getInstance(this)
        val current = manager.getCurrentVersion()

        if (current == null) {
            android.app.AlertDialog.Builder(this)
                .setTitle("No Version Active")
                .setMessage("No model version is currently loaded.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        if (current.isPinned) {
            // Unpin
            android.app.AlertDialog.Builder(this)
                .setTitle("Unpin Version?")
                .setMessage("Unpinning ${current.versionName} will allow automatic rollback if failures occur.\n\nContinue?")
                .setPositiveButton("Unpin") { _, _ ->
                    manager.unpinVersion()
                    android.widget.Toast.makeText(
                        this,
                        "Version unpinned",
                        android.widget.Toast.LENGTH_SHORT
                    ).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        } else {
            // Pin
            android.app.AlertDialog.Builder(this)
                .setTitle("Pin Version?")
                .setMessage("Pinning ${current.versionName} will prevent automatic rollback, even if failures occur.\n\nContinue?")
                .setPositiveButton("Pin") { _, _ ->
                    manager.pinCurrentVersion()
                    android.widget.Toast.makeText(
                        this,
                        "Version pinned",
                        android.widget.Toast.LENGTH_SHORT
                    ).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
    }

    private fun exportRollbackHistory() {
        val manager = ModelVersionManager.getInstance(this)
        val jsonData = manager.exportHistory()

        // Copy to clipboard
        val clipboard = getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
        val clip = android.content.ClipData.newPlainText("Rollback History", jsonData)
        clipboard.setPrimaryClip(clip)

        android.app.AlertDialog.Builder(this)
            .setTitle("💾 History Exported")
            .setMessage("Version history has been copied to clipboard as JSON.\n\nYou can paste it into a file or analysis tool.")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun resetRollbackData() {
        android.app.AlertDialog.Builder(this)
            .setTitle("Reset Rollback Data?")
            .setMessage("This will permanently delete:\n• All version history\n• Success/failure statistics\n• Rollback configuration\n\nThis cannot be undone. Continue?")
            .setPositiveButton("Reset") { _, _ ->
                val manager = ModelVersionManager.getInstance(this)
                manager.reset()

                android.widget.Toast.makeText(
                    this,
                    "Rollback data reset successfully",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
