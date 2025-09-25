// DOM Elements
const steps = document.querySelectorAll(".step");
const stepContents = document.querySelectorAll(".step-content");
const fileInput = document.getElementById("fileInput");
const chooseFileBtn = document.getElementById("chooseFileBtn");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const progressBar = document.getElementById("progressBar");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");
const loadingOverlay = document.getElementById("loadingOverlay");

// File handling elements
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFileBtn = document.getElementById("removeFile");

// Modal elements
const modal = document.getElementById("manualModal");
const closeModalBtn = document.getElementById("closeModal");
const manualCancelBtn = document.getElementById("manualCancelBtn");
const manualOkBtn = document.getElementById("manualOkBtn");

// Configuration
const CONFIG = {
    maxFileSize: 50 * 1024 * 1024, // 50MB
    allowedExtensions: ['.csv', '.xlsx', '.xls'],
    serverUrl: 'http://127.0.0.1:8000'
};

let currentStep = 0;
let currentFile = null;

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileExtension(filename) {
    return filename.toLowerCase().substring(filename.lastIndexOf('.'));
}

function showError(message, duration = 5000) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
    errorMessage.classList.add('show');
    
    setTimeout(() => {
        hideError();
    }, duration);
}

function hideError() {
    errorMessage.classList.remove('show');
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, 300);
}

function showLoading() {
    loadingOverlay.classList.remove('hidden');
    loadingOverlay.classList.add('show');
}

function hideLoading() {
    loadingOverlay.classList.remove('show');
    setTimeout(() => {
        loadingOverlay.classList.add('hidden');
    }, 300);
}

function updateProgress(step) {
    const progressPercentages = [25, 50, 75, 100];
    progressBar.style.width = progressPercentages[step] + '%';
}

function showStep(index) {
    // Update step indicators
    steps.forEach((step, i) => {
        step.classList.toggle("active", i === index);
    });

    // Show/hide step content
    stepContents.forEach((content, i) => {
        content.classList.toggle("hidden", i !== index);
    });

    // Update progress bar
    updateProgress(index);
    currentStep = index;
}

function showModal() {
    modal.classList.remove('hidden');
    setTimeout(() => {
        modal.classList.add('show');
    }, 10);
}

function hideModal() {
    modal.classList.remove('show');
    setTimeout(() => {
        modal.classList.add('hidden');
    }, 300);
}

function validateFile(file) {
    // Check file size
    if (file.size > CONFIG.maxFileSize) {
        throw new Error(`File size (${formatFileSize(file.size)}) exceeds the maximum allowed size of ${formatFileSize(CONFIG.maxFileSize)}`);
    }

    // Check file extension
    const extension = getFileExtension(file.name);
    if (!CONFIG.allowedExtensions.includes(extension)) {
        throw new Error(`File type ${extension} is not supported. Please upload a CSV or Excel file.`);
    }

    return true;
}

function displayFileInfo(file) {
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
}

function clearFileInfo() {
    fileInfo.classList.add('hidden');
    fileName.textContent = '';
    fileSize.textContent = '';
    currentFile = null;
    fileInput.value = '';
}

function showResults(data, mode) {
    const resultsContainer = document.getElementById("results");
    
    resultsContainer.innerHTML = `
        <div class="results-summary">
            <p class="success-message">✅ ${mode} Analysis Complete!</p>
            <p class="results-info">Your data has been processed and reports are ready for download.</p>
        </div>
        <div class="download-buttons">
            <a class="primary-btn" href="${CONFIG.serverUrl}${data.csv_url}" target="_blank">
                <span class="material-icons">file_download</span>
                Download Cleaned CSV
            </a>
            <a class="primary-btn" href="${CONFIG.serverUrl}${data.html_url}" target="_blank">
                <span class="material-icons">description</span>
                Download HTML Report
            </a>
            <a class="primary-btn" href="${CONFIG.serverUrl}${data.pdf_url}" target="_blank">
                <span class="material-icons">picture_as_pdf</span>
                Download PDF Report
            </a>
        </div>
    `;
}

function updateProcessingStatus(step) {
    const statusItems = document.querySelectorAll('.status-item');
    
    statusItems.forEach((item, index) => {
        if (index < step) {
            item.classList.remove('processing', 'pending');
            item.classList.add('completed');
        } else if (index === step) {
            item.classList.remove('pending', 'completed');
            item.classList.add('processing');
        } else {
            item.classList.remove('processing', 'completed');
            item.classList.add('pending');
        }
    });
}

async function makeRequest(url, formData) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Server error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Unable to connect to server. Please ensure the server is running.');
        }
        throw error;
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize
    showStep(currentStep);
    
    // File upload events
    chooseFileBtn.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            try {
                validateFile(file);
                currentFile = file;
                displayFileInfo(file);
                fileNameDisplay.textContent = file.name;
                
                // Auto-advance to mode selection
                setTimeout(() => {
                    showStep(1);
                }, 500);
            } catch (error) {
                showError(error.message);
                clearFileInfo();
            }
        }
    });

    // Drag and drop functionality
    const uploadBox = document.getElementById('upload-step');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadBox.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadBox.classList.add('dragover');
    }

    function unhighlight() {
        uploadBox.classList.remove('dragover');
    }

    uploadBox.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const file = files[0];
            try {
                validateFile(file);
                currentFile = file;
                fileInput.files = files; // Update file input
                displayFileInfo(file);
                fileNameDisplay.textContent = file.name;
                
                setTimeout(() => {
                    showStep(1);
                }, 500);
            } catch (error) {
                showError(error.message);
            }
        }
    }

    // Remove file button
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileInfo();
        showStep(0);
    });

    // AI Mode
    document.getElementById("aiButton").addEventListener("click", async function() {
        if (!currentFile) {
            showError("Please upload a file first");
            return;
        }

        showStep(2);
        showLoading();
        updateProcessingStatus(0);

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            setTimeout(() => updateProcessingStatus(1), 1000);
            
            const data = await makeRequest(`${CONFIG.serverUrl}/analyze`, formData);
            
            setTimeout(() => updateProcessingStatus(2), 500);
            
            hideLoading();
            showStep(3);
            showResults(data, "AI");
        } catch (error) {
            hideLoading();
            showError(`Analysis failed: ${error.message}`);
            showStep(1);
        }
    });

    // Manual Mode
    document.getElementById("manualButton").addEventListener("click", function() {
        if (!currentFile) {
            showError("Please upload a file first");
            return;
        }
        showModal();
    });

    // Modal event listeners
    closeModalBtn.addEventListener("click", hideModal);
    manualCancelBtn.addEventListener("click", hideModal);

    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            hideModal();
        }
    });

    // Escape key to close modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('show')) {
            hideModal();
        }
    });

    // Manual analysis confirmation
    manualOkBtn.addEventListener("click", async function() {
        const cleaning = document.getElementById("cleaningSelect").value;
        const analysis = document.getElementById("analysisSelect").value;

        if (!cleaning || !analysis) {
            showError("Please select both cleaning strategy and analysis type.");
            return;
        }

        if (!currentFile) {
            showError("Please upload a file first");
            return;
        }

        hideModal();
        showStep(2);
        showLoading();
        updateProcessingStatus(0);

        const formData = new FormData();
        formData.append("file", currentFile);
        formData.append("cleaning", cleaning);
        formData.append("analysis", analysis);

        try {
            setTimeout(() => updateProcessingStatus(1), 1000);
            
            const data = await makeRequest(`${CONFIG.serverUrl}/manual`, formData);
            
            setTimeout(() => updateProcessingStatus(2), 500);
            
            hideLoading();
            showStep(3);
            showResults(data, "Manual");
        } catch (error) {
            hideLoading();
            showError(`Analysis failed: ${error.message}`);
            showStep(1);
        }
    });

    // New analysis button
    document.addEventListener('click', function(e) {
        if (e.target && e.target.id === 'newAnalysis') {
            // Reset everything
            clearFileInfo();
            showStep(0);
            
            // Reset form selections
            document.getElementById("cleaningSelect").value = "";
            document.getElementById("analysisSelect").value = "";
            
            // Clear results
            document.getElementById("results").innerHTML = "";
        }
    });

    // Share results button
    document.addEventListener('click', function(e) {
        if (e.target && e.target.id === 'shareResults') {
            if (navigator.share) {
                navigator.share({
                    title: 'Data Analysis Results',
                    text: 'Check out my data analysis results!',
                    url: window.location.href
                });
            } else {
                // Fallback: copy to clipboard
                navigator.clipboard.writeText(window.location.href).then(() => {
                    showError('Results URL copied to clipboard!', 3000);
                });
            }
        }
    });

    // Error message dismiss
    document.getElementById('dismissError').addEventListener('click', hideError);

    // Form validation styling
    document.getElementById('cleaningSelect').addEventListener('change', function() {
        this.style.borderColor = this.value ? '#4CAF50' : '#e9ecef';
    });

    document.getElementById('analysisSelect').addEventListener('change', function() {
        this.style.borderColor = this.value ? '#4CAF50' : '#e9ecef';
    });

    // Step navigation (optional - allow clicking on steps)
    steps.forEach((step, index) => {
        step.addEventListener('click', () => {
            // Only allow going back to previous steps
            if (index < currentStep || (index === 1 && currentFile)) {
                showStep(index);
            }
        });
    });
});

// Global error handler
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showError('An unexpected error occurred. Please try again.');
});

window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showError('An unexpected error occurred. Please refresh the page and try again.');
});