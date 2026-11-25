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
-- Enhanced schema additions
create table if not exists price_bars_daily (
    ticker      text not null,
    date        date not null,
    open        numeric,
    high        numeric,
    low         numeric,
    close       numeric,
    adj_close   numeric,
    volume      bigint,
    source      text default 'yfinance',
    primary key (ticker, date)
);
create index if not exists idx_price_bars_daily_ticker_date on price_bars_daily (ticker, date);
create index if not exists idx_price_bars_daily_date on price_bars_daily (date);

create table if not exists position_performance_daily (
    id               bigserial primary key,
    account          text,
    ticker           text not null,
    date             date not null,
    shares           numeric default 0,
    price_close      numeric default 0,
    value_local      numeric default 0,
    value_cad        numeric default 0,
    cost_basis_cad   numeric default 0,
    unrealized_pnl_cad numeric default 0,
    realized_pnl_cad numeric default 0,
    day_pnl_cad      numeric default 0,
    weight           numeric default 0,
    unique (account, ticker, date)
);
create index if not exists idx_perf_daily_ticker_date on position_performance_daily (ticker, date);
create index if not exists idx_perf_daily_date on position_performance_daily (date);

create table if not exists recommendations (
    id           uuid primary key default gen_random_uuid(),
    run_id       bigint references brief_runs(id) on delete cascade,
    ticker       text not null,
    action       text,
    score        numeric,
    rationale    jsonb,
    constraints  jsonb,
    executed     boolean default false,
    outcome_1d   jsonb,
    outcome_7d   jsonb,
    outcome_30d  jsonb,
    created_at   timestamptz not null default now()
);
create index if not exists idx_recommendations_run_id on recommendations (run_id);
create index if not exists idx_recommendations_ticker on recommendations (ticker);

-- RLS for new tables
alter table price_bars_daily              enable row level security;
alter table position_performance_daily    enable row level security;
alter table recommendations               enable row level security;

create policy "allow read" on price_bars_daily for select using (true);
create policy "allow read" on position_performance_daily for select using (true);
create policy "allow read" on recommendations for select using (true);

create policy "allow write" on price_bars_daily
    for insert with check (true);
create policy "allow update" on price_bars_daily
    for update using (true) with check (true);

create policy "allow write" on position_performance_daily
    for insert with check (true);
create policy "allow update" on position_performance_daily
    for update using (true) with check (true);

create policy "allow write" on recommendations
    for insert with check (true);
create policy "allow update" on recommendations
    for update using (true) with check (true);

-- ---------------------------------------------------------------------------
-- Top-buys screen history (written by `main.py top-buys --persist`)
-- ---------------------------------------------------------------------------

create table if not exists top_buys_runs (
    id            bigserial primary key,
    run_date      date not null unique,
    generated_at  timestamptz not null default now(),
    universe      text,
    pick_limit    integer,
    min_analysts  integer default 3,
    weights_used  jsonb,
    picks_count   integer default 0,
    narrative     text
);

create table if not exists top_buys_picks (
    id               bigserial primary key,
    run_id           bigint not null references top_buys_runs(id) on delete cascade,
    rank             integer not null,
    symbol           text not null,
    company          text,
    sector           text,
    price            numeric,
    target           numeric,
    target_low       numeric,
    target_high      numeric,
    upside           numeric,
    score            numeric,
    verdict          text,
    tags             jsonb not null default '[]'::jsonb,
    score_reasons    jsonb not null default '[]'::jsonb,
    risk_flags       jsonb not null default '[]'::jsonb,
    analyst_score    numeric,
    quality_score    numeric,
    valuation_score  numeric,
    growth_score     numeric,
    momentum_score   numeric,
    underrated_score numeric,
    strong_buy       integer,
    buy              integer,
    hold             integer,
    sell             integer,
    strong_sell      integer,
    total_analysts   integer,
    data_coverage    numeric,
    risk_penalty     numeric,
    revisions        jsonb,
    unique (run_id, symbol)
);
create index if not exists idx_top_buys_picks_run on top_buys_picks (run_id, rank);
create index if not exists idx_top_buys_picks_symbol on top_buys_picks (symbol);

alter table top_buys_runs  enable row level security;
alter table top_buys_picks enable row level security;

create policy "allow read" on top_buys_runs  for select using (true);
create policy "allow read" on top_buys_picks for select using (true);

create policy "allow write" on top_buys_runs
    for insert with check (true);
create policy "allow update" on top_buys_runs
    for update using (true) with check (true);
create policy "allow delete" on top_buys_runs
    for delete using (true);

create policy "allow write" on top_buys_picks
    for insert with check (true);
create policy "allow update" on top_buys_picks
    for update using (true) with check (true);
create policy "allow delete" on top_buys_picks
    for delete using (true);

-- ---------------------------------------------------------------------------
-- Market-leaders / stock-status history (written by `main.py stock-status --persist`)
-- ---------------------------------------------------------------------------

create table if not exists stock_status_runs (
    id            bigserial primary key,
    run_date      date not null unique,
    generated_at  timestamptz not null default now(),
    universe      text,
    sort          text,
    us_count      integer default 0,
    ca_count      integer default 0,
    total_count   integer default 0,
    groups        jsonb not null default '[]'::jsonb
);

create table if not exists stock_status_rows (
    id             bigserial primary key,
    run_id         bigint not null references stock_status_runs(id) on delete cascade,
    group_name     text not null,
    rank           integer not null,
    symbol         text not null,
    company        text,
    currency       text,
    price          numeric,
    day_pct        numeric,
    market_cap     numeric,
    pe             numeric,
    fwd_pe         numeric,
    div_yield      numeric,
    high_52w       numeric,
    low_52w        numeric,
    ath            numeric,
    ath_date       text,
    from_ath_pct   numeric,
    target         numeric,
    upside_pct     numeric,
    strong_buy     integer,
    buy            integer,
    hold           integer,
    sell           integer,
    strong_sell    integer,
    total_analysts integer,
    consensus      text,
    unique (run_id, group_name, symbol)
);
create index if not exists idx_stock_status_rows_run on stock_status_rows (run_id, group_name, rank);
create index if not exists idx_stock_status_rows_symbol on stock_status_rows (symbol);

alter table stock_status_runs enable row level security;
alter table stock_status_rows enable row level security;

create policy "allow read" on stock_status_runs for select using (true);
create policy "allow read" on stock_status_rows for select using (true);

create policy "allow write" on stock_status_runs
    for insert with check (true);
create policy "allow update" on stock_status_runs
    for update using (true) with check (true);
create policy "allow delete" on stock_status_runs
    for delete using (true);

create policy "allow write" on stock_status_rows
    for insert with check (true);
create policy "allow update" on stock_status_rows
    for update using (true) with check (true);
create policy "allow delete" on stock_status_rows
    for delete using (true);
