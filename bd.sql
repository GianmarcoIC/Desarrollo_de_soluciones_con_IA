CREATE TABLE biblioteca (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(50) NOT NULL,
    detecciones JSONB,
    url TEXT NOT NULL,
    thumbnail_url TEXT NOT NULL,
    has_detection BOOLEAN NOT NULL,
    original_filename VARCHAR(255),
    source VARCHAR(50),
    confidence_average FLOAT,
    detection_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_biblioteca_timestamp ON biblioteca (timestamp);
CREATE INDEX idx_biblioteca_has_detection ON biblioteca (has_detection);


ALTER TABLE biblioteca DISABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow service role delete" ON biblioteca
    FOR DELETE TO service_role
    USING (true);