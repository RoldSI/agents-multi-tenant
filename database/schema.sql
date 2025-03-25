-- Schema for multi-tenant agent framework

-- Tools table
CREATE TABLE IF NOT EXISTS tools (
    tool_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    dependencies TEXT NOT NULL,  -- JSON array of dependencies
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLMs table
CREATE TABLE IF NOT EXISTS llms (
    llm_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); 