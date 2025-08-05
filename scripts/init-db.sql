-- Self-Healing Pipeline Guard - Database Initialization Script
-- This script sets up the initial database schema and data

\c healing_guard;

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create enum types
CREATE TYPE failure_status AS ENUM ('detected', 'analyzing', 'healing', 'resolved', 'failed');
CREATE TYPE healing_action_type AS ENUM ('retry', 'rollback', 'scale', 'restart', 'custom');
CREATE TYPE priority_level AS ENUM ('low', 'medium', 'high', 'critical');

-- Create sequences
CREATE SEQUENCE IF NOT EXISTS failure_id_seq;
CREATE SEQUENCE IF NOT EXISTS healing_attempt_id_seq;
CREATE SEQUENCE IF NOT EXISTS audit_log_id_seq;

-- Failures table
CREATE TABLE IF NOT EXISTS failures (
    id BIGINT PRIMARY KEY DEFAULT nextval('failure_id_seq'),
    failure_id UUID UNIQUE DEFAULT uuid_generate_v4(),
    repository VARCHAR(255) NOT NULL,
    branch VARCHAR(100) NOT NULL,
    commit_sha VARCHAR(40) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    failure_type VARCHAR(100) NOT NULL,
    status failure_status DEFAULT 'detected',
    priority priority_level DEFAULT 'medium',
    title TEXT NOT NULL,
    description TEXT,
    error_message TEXT,
    stack_trace TEXT,
    metadata JSONB DEFAULT '{}',
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Healing attempts table
CREATE TABLE IF NOT EXISTS healing_attempts (
    id BIGINT PRIMARY KEY DEFAULT nextval('healing_attempt_id_seq'),
    attempt_id UUID UNIQUE DEFAULT uuid_generate_v4(),
    failure_id UUID NOT NULL REFERENCES failures(failure_id) ON DELETE CASCADE,
    action_type healing_action_type NOT NULL,
    action_description TEXT,
    parameters JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    success BOOLEAN,
    error_message TEXT,
    logs TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Quantum planner tasks table
CREATE TABLE IF NOT EXISTS quantum_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 1,
    dependencies TEXT[] DEFAULT '{}',
    estimated_duration INTEGER, -- in seconds
    actual_duration INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    assigned_worker VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT PRIMARY KEY DEFAULT nextval('audit_log_id_seq'),
    event_id UUID UNIQUE DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(255),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- GDPR compliance tables
CREATE TABLE IF NOT EXISTS personal_data_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    record_id VARCHAR(255) UNIQUE NOT NULL,
    data_subject_id VARCHAR(255) NOT NULL,
    data_category VARCHAR(100) NOT NULL,
    processing_purpose VARCHAR(100) NOT NULL,
    data_fields TEXT[] DEFAULT '{}',
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    retention_period INTERVAL,
    consent_given BOOLEAN DEFAULT FALSE,
    consent_timestamp TIMESTAMP WITH TIME ZONE,
    anonymized BOOLEAN DEFAULT FALSE,
    deleted BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_subject_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    data_subject_id VARCHAR(255) NOT NULL,
    request_type VARCHAR(100) NOT NULL,
    requested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    completed_at TIMESTAMP WITH TIME ZONE,
    response_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User authentication and authorization tables
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_failures_repository_branch ON failures(repository, branch);
CREATE INDEX IF NOT EXISTS idx_failures_status ON failures(status);
CREATE INDEX IF NOT EXISTS idx_failures_detected_at ON failures(detected_at);
CREATE INDEX IF NOT EXISTS idx_failures_priority ON failures(priority);
CREATE INDEX IF NOT EXISTS idx_failures_metadata ON failures USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_healing_attempts_failure_id ON healing_attempts(failure_id);
CREATE INDEX IF NOT EXISTS idx_healing_attempts_status ON healing_attempts(status);
CREATE INDEX IF NOT EXISTS idx_healing_attempts_started_at ON healing_attempts(started_at);

CREATE INDEX IF NOT EXISTS idx_quantum_tasks_status ON quantum_tasks(status);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_priority ON quantum_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_created_at ON quantum_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_metadata ON quantum_tasks USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_timestamp ON performance_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_labels ON performance_metrics USING GIN(labels);

CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);

CREATE INDEX IF NOT EXISTS idx_personal_data_records_subject_id ON personal_data_records(data_subject_id);
CREATE INDEX IF NOT EXISTS idx_personal_data_records_category ON personal_data_records(data_category);
CREATE INDEX IF NOT EXISTS idx_personal_data_records_deleted ON personal_data_records(deleted);

CREATE INDEX IF NOT EXISTS idx_data_subject_requests_subject_id ON data_subject_requests(data_subject_id);
CREATE INDEX IF NOT EXISTS idx_data_subject_requests_status ON data_subject_requests(status);
CREATE INDEX IF NOT EXISTS idx_data_subject_requests_type ON data_subject_requests(request_type);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_failures_title_search ON failures USING GIN(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_failures_description_search ON failures USING GIN(to_tsvector('english', description));

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_failures_updated_at BEFORE UPDATE ON failures
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_healing_attempts_updated_at BEFORE UPDATE ON healing_attempts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_quantum_tasks_updated_at BEFORE UPDATE ON quantum_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_personal_data_records_updated_at BEFORE UPDATE ON personal_data_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_subject_requests_updated_at BEFORE UPDATE ON data_subject_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data
INSERT INTO users (username, email, password_hash, full_name, is_superuser) VALUES 
('admin', 'admin@terragonlabs.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj2.9xH8/CjG', 'System Administrator', true)
ON CONFLICT (username) DO NOTHING;

-- Create materialized views for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS failure_summary AS
SELECT 
    DATE_TRUNC('day', detected_at) as date,
    repository,
    failure_type,
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (COALESCE(resolved_at, CURRENT_TIMESTAMP) - detected_at))) as avg_resolution_time
FROM failures 
GROUP BY DATE_TRUNC('day', detected_at), repository, failure_type, status;

CREATE UNIQUE INDEX IF NOT EXISTS idx_failure_summary_unique ON failure_summary(date, repository, failure_type, status);

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_failure_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY failure_summary;
END;
$$ LANGUAGE plpgsql;

-- Create functions for common queries
CREATE OR REPLACE FUNCTION get_failure_statistics(
    start_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP - INTERVAL '30 days',
    end_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE(
    total_failures BIGINT,
    resolved_failures BIGINT,
    resolution_rate NUMERIC,
    avg_resolution_time NUMERIC,
    top_failure_types JSON
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'resolved') as resolved,
            AVG(EXTRACT(EPOCH FROM (resolved_at - detected_at))) FILTER (WHERE resolved_at IS NOT NULL) as avg_time
        FROM failures 
        WHERE detected_at BETWEEN start_date AND end_date
    ),
    top_types AS (
        SELECT json_agg(
            json_build_object(
                'failure_type', failure_type,
                'count', count
            ) ORDER BY count DESC
        ) as types
        FROM (
            SELECT failure_type, COUNT(*) as count
            FROM failures 
            WHERE detected_at BETWEEN start_date AND end_date
            GROUP BY failure_type
            ORDER BY count DESC
            LIMIT 10
        ) t
    )
    SELECT 
        s.total,
        s.resolved,
        CASE WHEN s.total > 0 THEN ROUND((s.resolved::NUMERIC / s.total::NUMERIC) * 100, 2) ELSE 0 END,
        ROUND(s.avg_time, 2),
        t.types
    FROM stats s, top_types t;
END;
$$ LANGUAGE plpgsql;

-- Create function for cleanup old data (GDPR compliance)
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Delete expired personal data records
    UPDATE personal_data_records 
    SET deleted = TRUE, data_fields = '{}' 
    WHERE retention_period IS NOT NULL 
    AND collected_at + retention_period < CURRENT_TIMESTAMP 
    AND NOT deleted;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old audit logs (keep for 2 years)
    DELETE FROM audit_logs 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '2 years';
    
    -- Delete old performance metrics (keep for 90 days)
    DELETE FROM performance_metrics 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Delete expired user sessions
    DELETE FROM user_sessions 
    WHERE expires_at < CURRENT_TIMESTAMP;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create cleanup job (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-expired-data', '0 2 * * *', 'SELECT cleanup_expired_data();');

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO healing_guard;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO healing_guard;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO healing_guard;

-- Create database health check function
CREATE OR REPLACE FUNCTION database_health_check()
RETURNS TABLE(
    status TEXT,
    details JSON
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'healthy'::TEXT,
        json_build_object(
            'timestamp', CURRENT_TIMESTAMP,
            'connections', (SELECT count(*) FROM pg_stat_activity),
            'database_size', pg_size_pretty(pg_database_size(current_database())),
            'version', version(),
            'uptime', EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - pg_postmaster_start_time()))
        );
EXCEPTION
    WHEN OTHERS THEN
        RETURN QUERY
        SELECT 
            'unhealthy'::TEXT,
            json_build_object(
                'error', SQLERRM,
                'timestamp', CURRENT_TIMESTAMP
            );
END;
$$ LANGUAGE plpgsql;

COMMIT;