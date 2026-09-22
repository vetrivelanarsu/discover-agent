alter table articles add column if not exists embedding vector(768);
alter table articles add column if not exists embedded_at timestamptz;

create index if not exists articles_embedding_idx
  on articles using hnsw (embedding vector_cosine_ops);