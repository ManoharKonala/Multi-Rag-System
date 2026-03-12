// UI Utility Functions
class UIManager {
    constructor() {
        this.notifications = [];
    }

    // Show notification
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icon = this.getNotificationIcon(type);
        notification.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;

        const container = document.getElementById('notifications');
        container.appendChild(notification);

        // Auto remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    if (notification.parentNode) {
                        container.removeChild(notification);
                    }
                }, 300);
            }
        }, duration);

        this.notifications.push(notification);
        return notification;
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    // Show loading state
    showLoading(message = 'Processing...') {
        const welcomeMessage = document.getElementById('welcomeMessage');
        const queryResults = document.getElementById('queryResults');
        const hybridResults = document.getElementById('hybridResults');
        const chatResults = document.getElementById('chatResults');
        const loadingIndicator = document.getElementById('loadingIndicator');

        // Hide all result sections
        welcomeMessage.style.display = 'none';
        queryResults.style.display = 'none';
        hybridResults.style.display = 'none';
        chatResults.style.display = 'none';

        // Show loading
        loadingIndicator.style.display = 'block';
        loadingIndicator.querySelector('p').textContent = message;
    }

    // Hide loading state
    hideLoading() {
        const loadingIndicator = document.getElementById('loadingIndicator');
        loadingIndicator.style.display = 'none';
    }

    // Show query results
    showQueryResults(data) {
        this.hideLoading();
        
        const welcomeMessage = document.getElementById('welcomeMessage');
        const queryResults = document.getElementById('queryResults');
        const hybridResults = document.getElementById('hybridResults');
        const chatResults = document.getElementById('chatResults');

        // Hide other sections
        welcomeMessage.style.display = 'none';
        hybridResults.style.display = 'none';
        chatResults.style.display = 'none';

        // Show query results
        queryResults.style.display = 'block';

        // Update AI response
        const aiResponse = document.getElementById('aiResponse');
        aiResponse.textContent = data.response || 'No response generated.';

        // Update sources
        this.renderSources(data.results || []);
    }

    // Show hybrid results
    showHybridResults(data) {
        this.hideLoading();
        
        const welcomeMessage = document.getElementById('welcomeMessage');
        const queryResults = document.getElementById('queryResults');
        const hybridResults = document.getElementById('hybridResults');
        const chatResults = document.getElementById('chatResults');

        // Hide other sections
        welcomeMessage.style.display = 'none';
        queryResults.style.display = 'none';
        chatResults.style.display = 'none';

        // Show hybrid results
        hybridResults.style.display = 'block';

        // Update counts
        const results = data.results || {};
        document.getElementById('textCount').textContent = (results.text || []).length;
        document.getElementById('imageCount').textContent = (results.image || []).length;
        document.getElementById('tableCount').textContent = (results.table || []).length;

        // Set active tab to text by default
        this.setActiveTab('text');
        this.renderTabContent('text', results.text || []);
    }

    // Show chat results
    showChatResults() {
        this.hideLoading();
        
        const welcomeMessage = document.getElementById('welcomeMessage');
        const queryResults = document.getElementById('queryResults');
        const hybridResults = document.getElementById('hybridResults');
        const chatResults = document.getElementById('chatResults');

        // Hide other sections
        welcomeMessage.style.display = 'none';
        queryResults.style.display = 'none';
        hybridResults.style.display = 'none';

        // Show chat results
        chatResults.style.display = 'block';
    }

    // Render sources
    renderSources(sources) {
        const container = document.getElementById('sourcesContainer');
        container.innerHTML = '';

        if (!sources || sources.length === 0) {
            container.innerHTML = '<p style="color: #718096; text-align: center;">No sources found.</p>';
            return;
        }

        sources.forEach((source, index) => {
            const sourceElement = document.createElement('div');
            sourceElement.className = 'source-item';
            
            const typeIcon = this.getContentTypeIcon(source.content_type);
            const score = (source.score * 100).toFixed(1);
            
            sourceElement.innerHTML = `
                <div class="source-header">
                    <div class="source-type">
                        <i class="${typeIcon}"></i>
                        ${source.content_type.charAt(0).toUpperCase() + source.content_type.slice(1)}
                    </div>
                    <div class="source-score">${score}% match</div>
                </div>
                <div class="source-content">${this.formatContent(source.content, source.content_type)}</div>
                <div class="source-meta">
                    <span>${source.source}</span>
                    <span>Page ${source.page_number}</span>
                </div>
            `;

            // Add click handler for tables
            if (source.content_type === 'table') {
                sourceElement.style.cursor = 'pointer';
                sourceElement.addEventListener('click', () => {
                    this.showTableDetails(source.id);
                });
            }

            container.appendChild(sourceElement);
        });
    }

    // Render tab content for hybrid results
    renderTabContent(contentType, results) {
        const container = document.getElementById('tabContent');
        container.innerHTML = '';

        if (!results || results.length === 0) {
            container.innerHTML = `<p style="color: #718096; text-align: center;">No ${contentType} results found.</p>`;
            return;
        }

        results.forEach((result, index) => {
            const resultElement = document.createElement('div');
            resultElement.className = 'source-item';
            
            const typeIcon = this.getContentTypeIcon(contentType);
            const score = (result.score * 100).toFixed(1);
            
            resultElement.innerHTML = `
                <div class="source-header">
                    <div class="source-type">
                        <i class="${typeIcon}"></i>
                        ${contentType.charAt(0).toUpperCase() + contentType.slice(1)}
                    </div>
                    <div class="source-score">${score}% match</div>
                </div>
                <div class="source-content">${this.formatContent(result.content, contentType)}</div>
                <div class="source-meta">
                    <span>${result.source}</span>
                    <span>Page ${result.page_number}</span>
                </div>
            `;

            container.appendChild(resultElement);
        });
    }

    // Set active tab
    setActiveTab(contentType) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-type="${contentType}"]`).classList.add('active');
    }

    // Get content type icon
    getContentTypeIcon(contentType) {
        const icons = {
            text: 'fas fa-file-text',
            image: 'fas fa-image',
            table: 'fas fa-table'
        };
        return icons[contentType] || 'fas fa-file';
    }

    // Format content based on type
    formatContent(content, contentType) {
        if (!content) return 'No content available';
        
        if (contentType === 'image') {
            return `<em>Image content: ${content}</em>`;
        } else if (contentType === 'table') {
            try {
                const tableData = JSON.parse(content);
                return tableData.description || tableData.title || 'Table data';
            } catch (e) {
                return content.substring(0, 200) + (content.length > 200 ? '...' : '');
            }
        } else {
            return content.substring(0, 300) + (content.length > 300 ? '...' : '');
        }
    }

    // Show table details
    async showTableDetails(tableId) {
        try {
            this.showNotification('Loading table data...', 'info', 2000);
            const tableData = await apiClient.getTableData(tableId);
            
            // Create a simple table view
            let tableHtml = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">';
            
            // Headers
            if (tableData.columns && tableData.columns.length > 0) {
                tableHtml += '<thead><tr>';
                tableData.columns.forEach(col => {
                    tableHtml += `<th style="border: 1px solid #e2e8f0; padding: 0.5rem; background: #f7fafc;">${col}</th>`;
                });
                tableHtml += '</tr></thead>';
            }
            
            // Data rows
            if (tableData.data && tableData.data.length > 0) {
                tableHtml += '<tbody>';
                tableData.data.slice(0, 10).forEach(row => { // Show first 10 rows
                    tableHtml += '<tr>';
                    if (Array.isArray(row)) {
                        row.forEach(cell => {
                            tableHtml += `<td style="border: 1px solid #e2e8f0; padding: 0.5rem;">${cell || ''}</td>`;
                        });
                    } else if (typeof row === 'object') {
                        tableData.columns.forEach(col => {
                            tableHtml += `<td style="border: 1px solid #e2e8f0; padding: 0.5rem;">${row[col] || ''}</td>`;
                        });
                    }
                    tableHtml += '</tr>';
                });
                tableHtml += '</tbody>';
            }
            
            tableHtml += '</table></div>';
            
            if (tableData.data && tableData.data.length > 10) {
                tableHtml += `<p style="margin-top: 1rem; color: #718096; font-size: 0.9rem;">Showing first 10 rows of ${tableData.data.length} total rows.</p>`;
            }
            
            // Show in modal
            this.showModal('Table Data', tableHtml);
            
        } catch (error) {
            this.showNotification('Failed to load table data: ' + error.message, 'error');
        }
    }

    // Show modal
    showModal(title, content) {
        const modal = document.getElementById('statsModal');
        const modalBody = document.getElementById('statsModalBody');
        
        modal.querySelector('.modal-header h3').innerHTML = `<i class="fas fa-table"></i> ${title}`;
        modalBody.innerHTML = content;
        modal.style.display = 'flex';
    }

    // Hide modal
    hideModal() {
        const modal = document.getElementById('statsModal');
        modal.style.display = 'none';
    }

    // Add chat message
    addChatMessage(message, isUser = false) {
        const container = document.getElementById('chatContainer');
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${isUser ? 'user' : 'assistant'}`;
        
        messageElement.innerHTML = `
            <div class="message-content">${message}</div>
        `;
        
        container.appendChild(messageElement);
        container.scrollTop = container.scrollHeight;
    }

    // Update system status
    updateSystemStatus(isHealthy, stats = {}) {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('span');
        
        if (isHealthy) {
            statusDot.className = 'status-dot';
            statusText.textContent = 'System Online';
        } else {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'System Offline';
        }
        
        // Update stats
        if (stats.total_documents !== undefined) {
            document.getElementById('docCount').textContent = stats.total_documents;
        }
        if (stats.total_elements !== undefined) {
            document.getElementById('elementCount').textContent = stats.total_elements;
        }
    }

    // Update upload progress
    updateUploadProgress(percentage) {
        const progressContainer = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        progressContainer.style.display = 'block';
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = `Uploading... ${Math.round(percentage)}%`;
        
        if (percentage >= 100) {
            progressText.textContent = 'Processing document...';
        }
    }

    // Hide upload progress
    hideUploadProgress() {
        const progressContainer = document.getElementById('uploadProgress');
        progressContainer.style.display = 'none';
    }

    // Show statistics modal
    async showStatistics() {
        try {
            const stats = await apiClient.getStatistics();
            
            let content = '<div style="display: grid; gap: 1rem;">';
            
            // Basic stats
            content += `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div style="text-align: center; padding: 1rem; background: #f7fafc; border-radius: 8px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">${stats.total_documents || 0}</div>
                        <div style="color: #718096;">Documents</div>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: #f7fafc; border-radius: 8px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">${stats.total_elements || 0}</div>
                        <div style="color: #718096;">Elements</div>
                    </div>
                </div>
            `;
            
            // Content type distribution
            if (stats.content_type_distribution) {
                content += '<h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Content Distribution</h4>';
                content += '<div style="display: grid; gap: 0.5rem;">';
                
                Object.entries(stats.content_type_distribution).forEach(([type, count]) => {
                    const percentage = stats.total_elements > 0 ? (count / stats.total_elements * 100).toFixed(1) : 0;
                    content += `
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #f7fafc; border-radius: 6px;">
                            <span style="text-transform: capitalize;">${type}</span>
                            <span style="font-weight: 600;">${count} (${percentage}%)</span>
                        </div>
                    `;
                });
                
                content += '</div>';
            }
            
            // Database health
            if (stats.database_health) {
                content += '<h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Database Status</h4>';
                content += '<div style="display: grid; gap: 0.5rem;">';
                
                Object.entries(stats.database_health).forEach(([db, status]) => {
                    const statusColor = status ? '#68d391' : '#fc8181';
                    const statusText = status ? 'Connected' : 'Disconnected';
                    content += `
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: #f7fafc; border-radius: 6px;">
                            <span style="text-transform: capitalize;">${db}</span>
                            <span style="color: ${statusColor}; font-weight: 600;">${statusText}</span>
                        </div>
                    `;
                });
                
                content += '</div>';
            }
            
            content += '</div>';
            
            this.showModal('System Statistics', content);
            
        } catch (error) {
            this.showNotification('Failed to load statistics: ' + error.message, 'error');
        }
    }
}

// Create global UI manager instance
const uiManager = new UIManager();

// Export for use in other modules
window.uiManager = uiManager;

