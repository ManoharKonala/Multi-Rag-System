-- Multi-RAG System Database Initialization Script

-- Create extension for UUID generation (required for uuid_generate_v4())
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_timestamp TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create document_elements table
CREATE TABLE IF NOT EXISTS document_elements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    element_type VARCHAR(50) NOT NULL, -- 'text', 'image', 'table'
    content TEXT,
    page_number INTEGER,
    position_x FLOAT,
    position_y FLOAT,
    width FLOAT,
    height FLOAT,
    embedding_id VARCHAR(255), -- Reference to vector database
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create queries table for analytics
CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_type VARCHAR(50),
    complexity VARCHAR(50),
    processing_time_ms INTEGER,
    confidence_score FLOAT,
    results_count INTEGER,
    user_session VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create query_results table
CREATE TABLE IF NOT EXISTS query_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
    element_id UUID REFERENCES document_elements(id) ON DELETE CASCADE,
    relevance_score FLOAT,
    rank_position INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create system_metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT,
    metric_unit VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    query_count INTEGER DEFAULT 0,
    upload_count INTEGER DEFAULT 0,
    metadata JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_upload_timestamp ON documents(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_document_elements_document_id ON document_elements(document_id);
CREATE INDEX IF NOT EXISTS idx_document_elements_type ON document_elements(element_type);
CREATE INDEX IF NOT EXISTS idx_document_elements_page ON document_elements(page_number);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_type ON queries(query_type);
CREATE INDEX IF NOT EXISTS idx_query_results_query_id ON query_results(query_id);
CREATE INDEX IF NOT EXISTS idx_query_results_score ON query_results(relevance_score);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_document_elements_metadata_gin ON document_elements USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_queries_metadata_gin ON queries USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_system_metrics_metadata_gin ON system_metrics USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_user_sessions_metadata_gin ON user_sessions USING GIN(metadata);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_elements_updated_at BEFORE UPDATE ON document_elements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for cleaning old data
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old queries and their results
    WITH deleted_queries AS (
        DELETE FROM queries 
        WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep
        RETURNING id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted_queries;
    
    -- Delete old system metrics
    DELETE FROM system_metrics 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    -- Delete inactive user sessions
    DELETE FROM user_sessions 
    WHERE last_activity < CURRENT_TIMESTAMP - INTERVAL '1 day' * 7; -- Keep sessions for 7 days
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create views for analytics
CREATE OR REPLACE VIEW query_analytics AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as query_count,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(confidence_score) as avg_confidence,
    AVG(results_count) as avg_results_count,
    query_type,
    complexity
FROM queries 
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp), query_type, complexity
ORDER BY hour DESC;

CREATE OR REPLACE VIEW document_analytics AS
SELECT 
    DATE_TRUNC('day', upload_timestamp) as day,
    COUNT(*) as documents_uploaded,
    SUM(file_size) as total_size,
    AVG(file_size) as avg_size,
    processing_status
FROM documents 
WHERE upload_timestamp >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', upload_timestamp), processing_status
ORDER BY day DESC;

CREATE OR REPLACE VIEW system_health AS
SELECT 
    metric_name,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    COUNT(*) as measurement_count
FROM system_metrics 
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY metric_name;

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit) VALUES
('system_startup', 1, 'boolean'),
('database_initialized', 1, 'boolean');

-- Grant permissions (adjust as needed for your security requirements)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO multirag_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO multirag_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO multirag_user;

