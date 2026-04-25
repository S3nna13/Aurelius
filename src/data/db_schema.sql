-- Aurelius LLM Platform — PostgreSQL + pgvector schema
-- 13 tables covering users, sessions, tasks, tools, memory, knowledge,
-- retrieval, evaluation, safety, and model configuration.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email      TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS tasks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID REFERENCES sessions(id),
    goal         TEXT NOT NULL,
    status       VARCHAR(32) DEFAULT 'pending',
    priority     INT DEFAULT 5,
    risk_profile VARCHAR(32) DEFAULT 'balanced',
    max_steps    INT DEFAULT 50,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS task_steps (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id     UUID REFERENCES tasks(id),
    step_number INT NOT NULL,
    action      JSONB,
    result      JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tools (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    schema      JSONB
);

CREATE TABLE IF NOT EXISTS tool_invocations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id     UUID REFERENCES tasks(id),
    tool_name   TEXT NOT NULL,
    args        JSONB,
    result      JSONB,
    duration_ms FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_items (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    content    TEXT NOT NULL,
    embedding  VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title      TEXT,
    content    TEXT NOT NULL,
    source     TEXT,
    embedding  VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_hits (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id     UUID REFERENCES tasks(id),
    document_id UUID REFERENCES knowledge_documents(id),
    similarity  FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id      UUID REFERENCES tasks(id),
    quality_score FLOAT,
    risk_class   INT,
    success      BOOLEAN,
    details      JSONB,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS safety_policies (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT UNIQUE NOT NULL,
    rules      JSONB,
    active     BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS safety_events (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id    UUID REFERENCES tasks(id),
    policy_id  UUID REFERENCES safety_policies(id),
    event_type TEXT,
    details    JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_configs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT UNIQUE NOT NULL,
    config     JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_memory_items_embedding
    ON memory_items USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_knowledge_docs_embedding
    ON knowledge_documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_tasks_status
    ON tasks (status);

CREATE INDEX IF NOT EXISTS idx_tasks_session
    ON tasks (session_id);
