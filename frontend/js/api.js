// API Configuration
const API_BASE_URL = 'http://localhost:5000';

class APIClient {
    constructor(baseUrl = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }

    // Health check
    async healthCheck() {
        return this.request('/health');
    }

    // Upload document
    async uploadDocument(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        
        return new Promise((resolve, reject) => {
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable && onProgress) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    onProgress(percentComplete);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (error) {
                        reject(new Error('Invalid JSON response'));
                    }
                } else {
                    try {
                        const errorResponse = JSON.parse(xhr.responseText);
                        reject(new Error(errorResponse.error || `Upload failed with status ${xhr.status}`));
                    } catch (error) {
                        reject(new Error(`Upload failed with status ${xhr.status}`));
                    }
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });

            xhr.open('POST', `${this.baseUrl}/upload`);
            xhr.send(formData);
        });
    }

    // Query documents
    async queryDocuments(query, options = {}) {
        const payload = {
            query,
            top_k: options.topK || 5,
            content_types: options.contentTypes || null
        };

        return this.request('/query', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    // Hybrid search
    async hybridSearch(query, topK = 5) {
        const payload = {
            query,
            top_k: topK
        };

        return this.request('/hybrid_search', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    // Get table data
    async getTableData(tableId) {
        return this.request(`/table/${tableId}`);
    }

    // Get statistics
    async getStatistics() {
        return this.request('/statistics');
    }

    // Clear all data
    async clearAllData() {
        return this.request('/clear_data', {
            method: 'POST'
        });
    }

    // Chat
    async chat(message, history = []) {
        const payload = {
            message,
            history
        };

        return this.request('/chat', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }
}

// Create global API client instance
const apiClient = new APIClient();

// Export for use in other modules
window.apiClient = apiClient;

