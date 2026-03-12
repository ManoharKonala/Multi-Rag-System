// Main Application Logic
class MultiRAGApp {
    constructor() {
        this.currentMode = 'query';
        this.chatHistory = [];
        this.uploadedFiles = [];
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.checkSystemHealth();
        this.startHealthCheckInterval();
    }

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('fileInput');
        const browseBtn = document.getElementById('browseBtn');
        const uploadArea = document.getElementById('uploadArea');

        browseBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            this.handleFileUpload(e.dataTransfer.files);
        });

        // Search
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('searchInput');

        searchBtn.addEventListener('click', () => this.handleSearch());
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSearch();
            }
        });

        // Mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setMode(e.target.closest('.mode-btn').dataset.mode);
            });
        });

        // Tab buttons for hybrid results
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const contentType = e.target.closest('.tab-btn').dataset.type;
                this.setActiveHybridTab(contentType);
            });
        });

        // Chat
        const chatSendBtn = document.getElementById('chatSendBtn');
        const chatInput = document.getElementById('chatInput');

        chatSendBtn.addEventListener('click', () => this.handleChatMessage());
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleChatMessage();
            }
        });

        // Header actions
        const statsBtn = document.getElementById('statsBtn');
        const clearBtn = document.getElementById('clearBtn');

        statsBtn.addEventListener('click', () => uiManager.showStatistics());
        clearBtn.addEventListener('click', () => this.handleClearData());

        // Modal close
        const closeStatsModal = document.getElementById('closeStatsModal');
        const statsModal = document.getElementById('statsModal');

        closeStatsModal.addEventListener('click', () => uiManager.hideModal());
        statsModal.addEventListener('click', (e) => {
            if (e.target === statsModal) {
                uiManager.hideModal();
            }
        });
    }

    async checkSystemHealth() {
        try {
            const health = await apiClient.healthCheck();
            uiManager.updateSystemStatus(true, health.statistics);
        } catch (error) {
            console.error('Health check failed:', error);
            uiManager.updateSystemStatus(false);
        }
    }

    startHealthCheckInterval() {
        // Check system health every 30 seconds
        setInterval(() => {
            this.checkSystemHealth();
        }, 30000);
    }

    async handleFileUpload(files) {
        const fileArray = Array.from(files);
        const pdfFiles = fileArray.filter(file => file.type === 'application/pdf');

        if (pdfFiles.length === 0) {
            uiManager.showNotification('Please select PDF files only.', 'warning');
            return;
        }

        for (const file of pdfFiles) {
            try {
                uiManager.showNotification(`Uploading ${file.name}...`, 'info', 2000);
                
                const response = await apiClient.uploadDocument(file, (progress) => {
                    uiManager.updateUploadProgress(progress);
                });

                uiManager.hideUploadProgress();
                uiManager.showNotification(`Successfully uploaded ${file.name}`, 'success');
                this.uploadedFiles.push(file.name);
                
                // Refresh system stats
                await this.checkSystemHealth();

            } catch (error) {
                uiManager.hideUploadProgress();
                uiManager.showNotification(`Failed to upload ${file.name}: ${error.message}`, 'error');
            }
        }

        // Clear file input
        document.getElementById('fileInput').value = '';
    }

    setMode(mode) {
        this.currentMode = mode;
        
        // Update mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');

        // Update search input placeholder
        const searchInput = document.getElementById('searchInput');
        const placeholders = {
            query: 'Ask a question about your documents...',
            hybrid: 'Search across all content types...',
            chat: 'Start a conversation...'
        };
        searchInput.placeholder = placeholders[mode] || placeholders.query;

        // Show appropriate results section
        if (mode === 'chat') {
            uiManager.showChatResults();
        }
    }

    async handleSearch() {
        const query = document.getElementById('searchInput').value.trim();
        
        if (!query) {
            uiManager.showNotification('Please enter a search query.', 'warning');
            return;
        }

        try {
            uiManager.showLoading('Searching documents...');

            if (this.currentMode === 'query') {
                await this.handleQueryMode(query);
            } else if (this.currentMode === 'hybrid') {
                await this.handleHybridMode(query);
            } else if (this.currentMode === 'chat') {
                await this.handleChatMode(query);
            }

        } catch (error) {
            uiManager.hideLoading();
            uiManager.showNotification(`Search failed: ${error.message}`, 'error');
        }
    }

    async handleQueryMode(query) {
        const contentTypes = this.getSelectedContentTypes();
        const topK = parseInt(document.getElementById('topK').value);

        const response = await apiClient.queryDocuments(query, {
            contentTypes: contentTypes.length > 0 ? contentTypes : null,
            topK
        });

        uiManager.showQueryResults(response);
        
        // Clear search input
        document.getElementById('searchInput').value = '';
    }

    async handleHybridMode(query) {
        const topK = parseInt(document.getElementById('topK').value);
        const response = await apiClient.hybridSearch(query, topK);

        uiManager.showHybridResults(response);
        
        // Clear search input
        document.getElementById('searchInput').value = '';
    }

    async handleChatMode(query) {
        // Add user message to chat
        uiManager.addChatMessage(query, true);
        
        // Clear input immediately
        document.getElementById('searchInput').value = '';

        try {
            const response = await apiClient.chat(query, this.chatHistory);
            
            // Add assistant response
            uiManager.addChatMessage(response.response, false);
            
            // Update chat history
            this.chatHistory.push({
                user: query,
                assistant: response.response
            });

            // Keep only last 10 exchanges
            if (this.chatHistory.length > 10) {
                this.chatHistory = this.chatHistory.slice(-10);
            }

        } catch (error) {
            uiManager.addChatMessage(`Sorry, I encountered an error: ${error.message}`, false);
        }
    }

    async handleChatMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();
        
        if (!message) return;

        // Add user message
        uiManager.addChatMessage(message, true);
        chatInput.value = '';

        try {
            const response = await apiClient.chat(message, this.chatHistory);
            
            // Add assistant response
            uiManager.addChatMessage(response.response, false);
            
            // Update chat history
            this.chatHistory.push({
                user: message,
                assistant: response.response
            });

            // Keep only last 10 exchanges
            if (this.chatHistory.length > 10) {
                this.chatHistory = this.chatHistory.slice(-10);
            }

        } catch (error) {
            uiManager.addChatMessage(`Sorry, I encountered an error: ${error.message}`, false);
        }
    }

    setActiveHybridTab(contentType) {
        uiManager.setActiveTab(contentType);
        
        // Get current hybrid results and render for this content type
        const query = document.getElementById('searchInput').value.trim();
        if (query) {
            // Re-render content for the selected tab
            // This would typically be stored from the last hybrid search
            // For now, we'll just update the UI
        }
    }

    getSelectedContentTypes() {
        const contentTypes = [];
        
        if (document.getElementById('searchText').checked) {
            contentTypes.push('text');
        }
        if (document.getElementById('searchImages').checked) {
            contentTypes.push('image');
        }
        if (document.getElementById('searchTables').checked) {
            contentTypes.push('table');
        }
        
        return contentTypes;
    }

    async handleClearData() {
        if (!confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
            return;
        }

        try {
            uiManager.showNotification('Clearing all data...', 'info', 2000);
            
            await apiClient.clearAllData();
            
            // Reset application state
            this.chatHistory = [];
            this.uploadedFiles = [];
            
            // Clear chat container
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.innerHTML = '';
            
            // Show welcome message
            const welcomeMessage = document.getElementById('welcomeMessage');
            const queryResults = document.getElementById('queryResults');
            const hybridResults = document.getElementById('hybridResults');
            const chatResults = document.getElementById('chatResults');
            
            welcomeMessage.style.display = 'block';
            queryResults.style.display = 'none';
            hybridResults.style.display = 'none';
            chatResults.style.display = 'none';
            
            // Refresh system stats
            await this.checkSystemHealth();
            
            uiManager.showNotification('All data cleared successfully.', 'success');
            
        } catch (error) {
            uiManager.showNotification(`Failed to clear data: ${error.message}`, 'error');
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MultiRAGApp();
});

// Handle tab switching for hybrid results
document.addEventListener('click', (e) => {
    if (e.target.closest('.tab-btn')) {
        const contentType = e.target.closest('.tab-btn').dataset.type;
        
        // Get the last hybrid search results from a global variable or re-fetch
        // For now, we'll just switch the tab visually
        uiManager.setActiveTab(contentType);
        
        // In a real implementation, you'd store the last search results
        // and re-render the content for the selected tab
    }
});

// Add CSS for slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

