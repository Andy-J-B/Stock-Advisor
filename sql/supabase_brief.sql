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

-- Richer per-ticker enrichment captured with each run (best-effort columns).
alter table ticker_scores add column if not exists price numeric;
alter table ticker_scores add column if not exists day_change_pct numeric;
alter table ticker_scores add column if not exists top_headline text;
alter table ticker_scores add column if not exists analyst_breakdown text;
alter table ticker_scores add column if not exists signal_agreement text;
alter table ticker_scores add column if not exists anomaly_detail text;
alter table ticker_scores add column if not exists portfolio_weight numeric;
alter table ticker_scores add column if not exists recommendation text;

-- Create index so the dashboard's per-ticker history lookup is fast.
create index if not exists idx_ticker_scores_ticker_run on ticker_scores (ticker, run_id);

-- Live watchlist: mirrored from the local SQLite watchlist so the nightly
-- CI brief can see current trackers even on a fresh runner.
create table if not exists watchlist (
    ticker    text primary key,
    added_at  timestamptz not null default now()
);

-- Non-stock market context stored per run: major index moves + top news,
-- so the dashboard and webhook can recap the day beyond the tickers.
create table if not exists market_overview (
    run_date     date primary key,
    indices      jsonb not null,
    news         jsonb not null,
    generated_at timestamptz not null default now()
);

-- Portfolio data: normalized, persistent copy of the local portfolio so the
-- nightly CI brief + newsletter can render portfolio status without a local
-- database.  Three tables:
--   * portfolio_holdings        — current positions (portfolio of record)
--   * portfolio_snapshots       — one immutable aggregate row per brief run
--   * portfolio_snapshot_items  — per-holding breakdown of each snapshot
--
-- This replaces the original single JSONB-blob `portfolio_snapshots` design
-- (dropped below) with a queryable relational schema.

-- One-time migration: drop only the *legacy* JSONB portfolio_snapshots table
-- (it had a `data` jsonb column). Once the normalized schema is in place no
-- snapshot history is ever wiped by re-running this file.
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public'
      and table_name = 'portfolio_snapshots'
      and column_name = 'data'
  ) then
    drop table portfolio_snapshots cascade;
  end if;
end $$;

-- Make this file re-runnable: clear every existing policy in public so the
-- create policy statements below can be re-applied fresh.
do $$
declare r record;
begin
  for r in
    select policyname as name, tablename as tbl
    from pg_policies
    where schemaname = 'public'
  loop
    execute format('drop policy if exists %I on %I', r.name, r.tbl);
  end loop;
end $$;

create table if not exists portfolio_holdings (
    id         bigserial primary key,
    account    text not null,
    ticker     text not null,
    shares     numeric not null default 0,
    avg_price  numeric not null default 0,
    updated_at timestamptz not null default now(),
    unique (account, ticker)
);
create index if not exists idx_portfolio_holdings_ticker on portfolio_holdings (ticker);

-- Account cash/initial balances (in the account's native currency, mirroring
-- the local Account table) — lets a CI brief rebuild the exact net worth even
-- for accounts with no positions.
create table if not exists portfolio_accounts (
    id            bigserial primary key,
    account       text not null unique,
    cash          numeric not null default 0,
    initial_cash  numeric not null default 0,
    updated_at    timestamptz not null default now()
);

create table if not exists portfolio_snapshots (
    id             bigserial primary key,
    run_id         bigint not null references brief_runs(id) on delete cascade unique,
    run_date       date not null unique,
    net_worth_cad  numeric not null,
    invested_cad   numeric not null,
    cash_cad       numeric not null default 0,
    cost_cad       numeric not null default 0,
    day_change_cad numeric not null default 0,
    day_pct        numeric not null default 0,
    return_pct     numeric not null default 0,
    return_cad     numeric not null default 0,
    all_time_pct   numeric not null default 0,
    all_time_cad   numeric not null default 0,
    fx_usd_cad     numeric not null default 1,
    generated_at   timestamptz not null default now()
);

create table if not exists portfolio_snapshot_items (
    id             bigserial primary key,
    snapshot_id    bigint not null references portfolio_snapshots(id) on delete cascade,
    ticker         text not null,
    account        text not null,
    shares         numeric not null,
    avg_price      numeric not null default 0,
    price          numeric,
    day_change_pct numeric,
    day_change_cad numeric not null default 0,
    value_cad      numeric not null default 0,
    return_pct     numeric not null default 0,
    return_cad     numeric not null default 0,
    unique (snapshot_id, account, ticker)
);
create index if not exists idx_snapshot_items_ticker on portfolio_snapshot_items (ticker);

-- Heal migrations: if a previous destructive run dropped the parent table via
-- CASCADE, the items/unique FK constraints can be missing. Re-create them so
-- PostgREST upserts and cascades keep working.
do $$
begin
  if not exists (
    select 1 from pg_constraint c
    join pg_class t on t.oid = c.conrelid
    where t.relname = 'portfolio_snapshot_items' and c.contype = 'f'
  ) then
    alter table portfolio_snapshot_items
      add constraint portfolio_snapshot_items_snapshot_id_fkey
      foreign key (snapshot_id) references portfolio_snapshots(id) on delete cascade;
  end if;
end $$;

-- Row-level security: the anon key (used by both the nightly `brief --persist`
-- in CI and the read-only Streamlit dashboard) needs select + write access.
alter table brief_runs         enable row level security;
alter table ticker_scores      enable row level security;
alter table watchlist          enable row level security;
alter table market_overview    enable row level security;
alter table portfolio_snapshots      enable row level security;
alter table portfolio_snapshot_items enable row level security;
alter table portfolio_holdings        enable row level security;
alter table portfolio_accounts        enable row level security;

create policy "allow read" on brief_runs            for select using (true);
create policy "allow read" on ticker_scores         for select using (true);
create policy "allow read" on watchlist             for select using (true);
create policy "allow read" on market_overview       for select using (true);
create policy "allow read" on portfolio_snapshots        for select using (true);
create policy "allow read" on portfolio_snapshot_items   for select using (true);
create policy "allow read" on portfolio_holdings          for select using (true);
create policy "allow read" on portfolio_accounts          for select using (true);

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

create policy "allow write" on market_overview
    for insert with check (true);
create policy "allow update" on market_overview
    for update using (true) with check (true);

create policy "allow write" on portfolio_snapshots
    for insert with check (true);
create policy "allow update" on portfolio_snapshots
    for update using (true) with check (true);

create policy "allow write" on portfolio_snapshot_items
    for insert with check (true);
create policy "allow update" on portfolio_snapshot_items
    for update using (true) with check (true);
create policy "allow delete" on portfolio_snapshot_items
    for delete using (true);

create policy "allow write" on portfolio_holdings
    for insert with check (true);
create policy "allow update" on portfolio_holdings
    for update using (true) with check (true);
create policy "allow delete" on portfolio_holdings
    for delete using (true);

create policy "allow write" on portfolio_accounts
    for insert with check (true);
create policy "allow update" on portfolio_accounts
    for update using (true) with check (true);
create policy "allow delete" on portfolio_accounts
    for delete using (true);