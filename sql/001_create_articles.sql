create extension if not exists vector;

create table if not exists articles (
  id bigserial primary key,
  url text unique not null,
  title text not null,
  section text,
  author text,
  published_at timestamptz,
  body text,
  fetched_at timestamptz default now()
);