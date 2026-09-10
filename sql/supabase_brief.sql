-- Supabase / Postgres schema for the Nightly Brief history
-- Run this in the Supabase SQL editor (or psql) once to set up the tables
-- the `main.py brief --persist` command writes to.

create table if not exists brief_runs (
    id            bigserial primary key,
    run_date      date not null unique,
    generated_at  timestamptz not null default now(),
    weights_used  jsonb not null,
    tickers       jsonb
);

-- Schema evolution: make sure existing brief_runs tables get the tickers
-- column (the nightly CI brief reads the last run's universe from here).
alter table brief_runs add column if not exists tickers jsonb;

create table if not exists ticker_scores (
    id             bigserial primary key,
    run_id         bigint not null references brief_runs(id) on delete cascade,
    ticker         text not null,
    composite      numeric not null,
    sentiment      numeric,
    technical      numeric,
    ml_pred        numeric,
    analyst        numeric,
    anomaly_flag   boolean not null default false,
    reasoning      text,
    unique (run_id, ticker)
);

-- Create index so the dashboard's per-ticker history lookup is fast.
create index if not exists idx_ticker_scores_ticker_run on ticker_scores (ticker, run_id);

-- Live watchlist: mirrored from the local SQLite watchlist so the nightly
-- CI brief can see current trackers even on a fresh runner.
create table if not exists watchlist (
    ticker    text primary key,
    added_at  timestamptz not null default now()
);

-- Row-level security: the anon key (used by both the nightly `brief --persist`
-- in CI and the read-only Streamlit dashboard) needs select + write access.
alter table brief_runs    enable row level security;
alter table ticker_scores enable row level security;
alter table watchlist     enable row level security;

create policy "allow read" on brief_runs   for select using (true);
create policy "allow read" on ticker_scores for select using (true);
create policy "allow read" on watchlist    for select using (true);

create policy "allow write" on brief_runs
    for insert with check (true);
create policy "allow update" on brief_runs
    for update using (true) with check (true);

create policy "allow write" on ticker_scores
    for insert with check (true);
create policy "allow update" on ticker_scores
    for update using (true) with check (true);

create policy "allow write" on watchlist
    for insert with check (true);
create policy "allow update" on watchlist
    for update using (true) with check (true);
create policy "allow delete" on watchlist
    for delete using (true);