# Database Schema Migration Plan — Lot-Level Ledger (v2)

**Audience:** AI coding agent implementing the change.
**Status:** Design specification — not yet implemented.
**Scope:** Local SQLite database, `src/` domain layer, CLI, tests, and the Supabase mirror.

---

## 0. TL;DR

Replace the mutable “current position” model with an **append-only transaction ledger + FIFO tax lots**:

- `Account.cash` and `Holding.shares/avg_price` stop being the source of truth.
- Every deposit, withdrawal, buy, sell, dividend, fee and adjustment becomes a dated `txn` row.
- Every buy creates a `lot`; every sell consumes lots (FIFO by default) and writes `lot_disposal` rows with realized P&L.
- Cash and positions are **derived** from the ledger, then optionally cached for display.
- Keep `portfolio.load()` and `portfolio.snapshot()` as compatibility facades so the CLI, brief, newsletter and chat advisor keep working while the internals change.
- Keep SQLite + Peewee, but add **versioned migrations** (`playhouse.migrate` + `schema_version`). Do not introduce SQLAlchemy/Alembic in this change.
- Store money as **integer cents** and share quantities as **integer micro-shares** in SQLite. Convert to `Decimal`/`float` only at the UI and analytics boundaries.
- Backfill existing holdings into one **estimated** lot per position, then reconcile before deleting legacy tables.
- Mirror the ledger to Supabase so history is queryable in CI and the dashboard.

The result: individual purchases are recorded with their date, quantity, price, currency, fee and FX rate; realized gains, dividends and cash flow become auditable.

---

## 1. Current state audit

### 1.1 Existing models (`src/database.py`)

| Model | Table | Used? | Problem |
|---|---|---|---|
| `Account` | `account` | Yes | `cash` is a mutable scalar; no cash history. `initial_cash` is a manually overwritten baseline. |
| `Holding` | `holding` | Yes | Only the current aggregate (`shares`, `avg_price`). Overwritten on every buy/sell. No purchase date, lot, fee or currency. |
| `Transaction` | `transaction` | **No** | Model exists but is never written by `portfolio.py`. |
| `PositionLot` | `positionlot` | **Only migration** | Created by `_migrate_holdings_to_lots()`, never updated by buys/sells. |
| `PriceBar` | `pricebar` | Rarely | Keyed by ticker text, not an instrument identity. |
| `CorporateAction` | `corporateaction` | **No** | No split/merger handling. |
| `Dividend` | `dividend` | **No** | Dividends are never recorded. |
| `RealizedPnL` | `realizedpnl` | **No** | Realized gains are never recorded. |
| `PositionPerformanceDaily` | `positionperformancedaily` | **No** | No daily position snapshots are written. |
| `NetWorthSnapshot` | `networthsnapshot` | Yes | Aggregate only; no per-position history. |
| `FxRate` | `fxrate` | **No** | FX is fetched live and not persisted with transactions. |
| `Recommendation` | `recommendation` | Partly | Not linked to an instrument. |
| `Setting`, `Watchlist`, `CacheEntry` | — | Yes | Watchlist is ticker text, not an instrument FK. |

### 1.2 Concrete bugs and inconsistencies

1. **No purchase history.** `add_position()` averages into `Holding.avg_price` and discards the original date and lot.
2. **`_migrate_holdings_to_lots()` is defined twice** in `src/database.py` (the second definition silently wins).
3. **`PositionLot` is dead code** after migration; buys/sells never touch it.
4. **`Transaction` is never written**, so there is no cash or trade audit trail.
5. **`sell_position()` always credits the CAD account**, even when selling from the USD account, and never records realized P&L or consumes lots.
6. **`remove_position()` deletes shares without proceeds, cash flow or realized P&L.**
7. **`update_cash()` overwrites the balance** with no record of why it changed.
8. **`set_initial_cash()` overwrites the all-time baseline** with no history.
9. **Currency ambiguity:** `add_position(..., currency="USD")` converts the cash debit but stores `avg_price` unconverted. `view_portfolio()` then treats `avg_price` as the account currency and only applies FX for the whole USD account, so a US stock held in a CAD account is valued incorrectly.
10. **`save()` can delete holdings** and bypass any lot accounting.
11. **`setup.py` still creates `portfolio.json` / `settings.json`**, even though SQLite is the store of record.
12. **No schema versioning.** `db.create_tables(safe=True)` cannot add/rename columns, so schema changes are ad-hoc.
13. **Supabase only mirrors current holdings**, so CI and the dashboard have no transaction or realized-P&L history.
14. **Floats for money** (`FloatField`) cause rounding drift across repeated buys/sells.

### 1.3 What the user needs

- “I bought 10 shares of AAPL on 2026-03-14 at $172.40 with a $4.95 fee.”
- “I sold 4 of those shares on 2026-09-02 at $210.10; what is my realized gain?”
- “What is my remaining cost basis per lot?”
- “How much did I deposit/withdraw over time?”
- “What dividends did I receive?”
- “What was my position and P&L on a past date?”
- “Did a split change my share count and cost basis correctly?”

---

## 2. Design decisions

### 2.1 Keep SQLite + Peewee for the local store

| Option | Verdict | Why |
|---|---|---|
| **SQLite + Peewee + `playhouse.migrate`** | **Recommended** | Zero new runtime dependencies, already used everywhere, single-user local app, transactional, easy backup, SQLite is sufficient for this data volume. |
| SQLAlchemy + Alembic | Not now | More standard migrations, but requires rewriting every Peewee model, repository and test fixture. Only worth it if Postgres becomes the primary store or the app becomes multi-user. The dashboard already uses SQLAlchemy read-only against Postgres, which is fine. |
| Raw SQL migration files only | Partial | Good for data backfills, but no model/DDL coupling. Use raw SQL inside Peewee migrations where `playhouse.migrate` cannot express a rebuild. |
| DuckDB / Postgres locally | No | Analytical/OLTP mismatch or unnecessary server dependency. |

**Decision:** SQLite remains the operational database. Add a small migration package under `src/migrations/` with a `schema_version` table and ordered `up()`/`down()` functions. Use `playhouse.migrate.SqliteMigrator` for simple `ALTER TABLE` operations and raw SQL for table rebuilds and data backfills.

### 2.2 Append-only ledger, derived state

The ledger is the source of truth. Positions and cash are views/queries over it:

```
txn (cash + trade events) ──► lot (remaining cost basis) ──► v_position
        │                          │
        └──► v_account_cash        └──► lot_disposal ──► realized P&L
```

`portfolio.load()` may cache/denormalize for the CLI, but it must be reproducible from the ledger.

### 2.3 Money and quantity representation

- **Money:** integer **cents** in SQLite (`*_cents`). Use `Decimal` in Python and convert at repository boundaries. Never compare floats for money.
- **Quantity:** integer **micro-shares** (`*_micros`, 1 share = 1_000_000). Supports fractional shares without float drift.
- **FX:** integer **e-8 rate** (`fx_rate_e8`, 1.0 = 100_000_000). Example: USD→CAD 1.3542 = `135_420_000`.
- **Percentages/weights:** integer e-6 (`weight_e6`) for stored analytics; display converts.
- **Prices in `price_bar`:** may stay `REAL` because they feed pandas/analytics, not the ledger.
- **Postgres mirror:** use `NUMERIC(18,2)` for money, `NUMERIC(24,8)` for quantities, `NUMERIC(18,8)` for FX.

Create `src/money.py`:

```python
from decimal import Decimal, ROUND_HALF_UP

MICROS = Decimal("0.000001")

def to_cents(value) -> int:
    return int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(Decimal("0.01"))

def to_micros(value) -> int:
    return int((Decimal(str(value)) / MICROS).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def from_micros(micros: int) -> Decimal:
    return (Decimal(micros) * MICROS).quantize(MICROS)

def to_e8(value) -> int:
    return int((Decimal(str(value)) * 100_000_000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def from_e8(value: int) -> Decimal:
    return (Decimal(value) / 100_000_000).quantize(Decimal("0.00000001"))
```

Do **not** rewrite pandas/ML/optimizer code to `Decimal`. Convert only at the persistence boundary.

### 2.4 Cost basis method

- Default: **FIFO** (consume the oldest open lot first).
- Make the method configurable per account (`account.cost_basis_method` = `fifo`, `lifo`, `average`), defaulting to `fifo`.
- Canadian tax reporting often uses average cost for identical properties; the ledger still stores lots, and a report can aggregate them. Do not destroy lot detail.
- Superficial-loss and wash-sale rules are **out of scope** for this migration; note them as future work.

---

## 3. Target schema

### 3.1 Entity relationships

```
account 1──* txn *──1 instrument
account 1──* lot *──1 instrument
lot     1──* lot_disposal *──1 txn (sell)
account 1──* dividend *──1 instrument
instrument 1──* corporate_action
instrument 1──* price_bar
account 1──* portfolio_snapshot 1──* position_snapshot
```

### 3.2 SQLite DDL

Use ISO-8601 `TEXT` dates (`YYYY-MM-DD`) and UTC timestamps (`YYYY-MM-DDTHH:MM:SSZ`). Enable foreign keys and WAL mode in `init_db()`.

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER PRIMARY KEY,
    applied_at   TEXT NOT NULL,
    description  TEXT NOT NULL,
    checksum     TEXT
);

CREATE TABLE IF NOT EXISTS account (
    id                    INTEGER PRIMARY KEY,
    name                  TEXT NOT NULL UNIQUE,
    currency              TEXT NOT NULL DEFAULT 'CAD',
    kind                  TEXT NOT NULL DEFAULT 'brokerage',
    cost_basis_method     TEXT NOT NULL DEFAULT 'fifo',
    initial_cash_cents    INTEGER NOT NULL DEFAULT 0,  -- legacy all-time baseline
    opened_at             TEXT,
    is_active             INTEGER NOT NULL DEFAULT 1,
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS instrument (
    id                    INTEGER PRIMARY KEY,
    symbol                TEXT NOT NULL,
    exchange              TEXT NOT NULL DEFAULT '',
    currency              TEXT NOT NULL,
    name                  TEXT,
    asset_type            TEXT NOT NULL DEFAULT 'equity',
    us_equivalent_symbol  TEXT,
    is_active             INTEGER NOT NULL DEFAULT 1,
    created_at            TEXT NOT NULL,
    UNIQUE (symbol, exchange)
);
CREATE INDEX IF NOT EXISTS idx_instrument_symbol ON instrument (symbol);

CREATE TABLE IF NOT EXISTS txn (
    id                  INTEGER PRIMARY KEY,
    account_id          INTEGER NOT NULL REFERENCES account(id) ON DELETE CASCADE,
    instrument_id       INTEGER REFERENCES instrument(id) ON DELETE RESTRICT,
    type                TEXT NOT NULL CHECK (type IN (
                            'opening', 'deposit', 'withdrawal', 'buy', 'sell',
                            'dividend', 'fee', 'interest', 'transfer_in',
                            'transfer_out', 'adjustment', 'split'
                        )),
    trade_date          TEXT NOT NULL,
    settle_date         TEXT,
    quantity_micros     INTEGER NOT NULL DEFAULT 0,
    price_cents         INTEGER NOT NULL DEFAULT 0,   -- instrument currency
    gross_cents         INTEGER NOT NULL DEFAULT 0,   -- instrument currency
    fee_cents           INTEGER NOT NULL DEFAULT 0,   -- account currency
    currency            TEXT NOT NULL,                -- txn/price currency
    fx_rate_e8          INTEGER NOT NULL DEFAULT 100000000, -- account per instrument
    cash_delta_cents    INTEGER NOT NULL DEFAULT 0,   -- signed, account currency
    transfer_group_id   TEXT,
    external_id         TEXT,
    note                TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE (account_id, external_id)
);
CREATE INDEX IF NOT EXISTS idx_txn_account_date ON txn (account_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_txn_instrument_date ON txn (instrument_id, trade_date);
CREATE INDEX IF NOT EXISTS idx_txn_type_date ON txn (type, trade_date);

CREATE TABLE IF NOT EXISTS lot (
    id                                  INTEGER PRIMARY KEY,
    account_id                          INTEGER NOT NULL REFERENCES account(id) ON DELETE CASCADE,
    instrument_id                       INTEGER NOT NULL REFERENCES instrument(id) ON DELETE RESTRICT,
    open_txn_id                         INTEGER NOT NULL REFERENCES txn(id) ON DELETE CASCADE,
    acquired_date                       TEXT NOT NULL,
    original_quantity_micros            INTEGER NOT NULL,
    remaining_quantity_micros           INTEGER NOT NULL,
    cost_per_share_cents                INTEGER NOT NULL,   -- instrument currency, derived
    cost_currency                       TEXT NOT NULL,
    original_cost_basis_cents           INTEGER NOT NULL,   -- instrument currency
    remaining_cost_basis_cents          INTEGER NOT NULL,   -- instrument currency
    original_cost_basis_account_cents   INTEGER NOT NULL,   -- account currency
    remaining_cost_basis_account_cents  INTEGER NOT NULL,   -- account currency
    fx_rate_e8                          INTEGER NOT NULL,
    fee_cents_account                   INTEGER NOT NULL DEFAULT 0,
    source                              TEXT NOT NULL DEFAULT 'buy',
    is_estimated                        INTEGER NOT NULL DEFAULT 0,
    closed_at                           TEXT,
    UNIQUE (open_txn_id)
);
CREATE INDEX IF NOT EXISTS idx_lot_position
    ON lot (account_id, instrument_id, remaining_quantity_micros);
CREATE INDEX IF NOT EXISTS idx_lot_acquired
    ON lot (account_id, instrument_id, acquired_date);

CREATE TABLE IF NOT EXISTS lot_disposal (
    id                              INTEGER PRIMARY KEY,
    lot_id                          INTEGER NOT NULL REFERENCES lot(id) ON DELETE CASCADE,
    sell_txn_id                     INTEGER NOT NULL REFERENCES txn(id) ON DELETE CASCADE,
    account_id                      INTEGER NOT NULL REFERENCES account(id) ON DELETE CASCADE,
    instrument_id                   INTEGER NOT NULL REFERENCES instrument(id) ON DELETE RESTRICT,
    quantity_micros                 INTEGER NOT NULL,
    proceeds_cents                  INTEGER NOT NULL,   -- account currency
    cost_basis_cents                INTEGER NOT NULL,   -- account currency
    realized_pnl_cents              INTEGER NOT NULL,   -- account currency
    proceeds_instrument_cents       INTEGER NOT NULL,
    cost_basis_instrument_cents     INTEGER NOT NULL,
    realized_pnl_instrument_cents   INTEGER NOT NULL,
    currency                        TEXT NOT NULL,
    fx_rate_e8                      INTEGER NOT NULL,
    realized_at                     TEXT NOT NULL,
    UNIQUE (lot_id, sell_txn_id)
);
CREATE INDEX IF NOT EXISTS idx_disposal_sell ON lot_disposal (sell_txn_id);
CREATE INDEX IF NOT EXISTS idx_disposal_instrument_date
    ON lot_disposal (instrument_id, realized_at);

CREATE TABLE IF NOT EXISTS dividend (
    id                      INTEGER PRIMARY KEY,
    account_id              INTEGER NOT NULL REFERENCES account(id) ON DELETE CASCADE,
    instrument_id           INTEGER NOT NULL REFERENCES instrument(id) ON DELETE RESTRICT,
    txn_id                  INTEGER NOT NULL UNIQUE REFERENCES txn(id) ON DELETE CASCADE,
    ex_date                 TEXT,
    pay_date                TEXT NOT NULL,
    shares_micros           INTEGER NOT NULL,
    amount_per_share_cents  INTEGER NOT NULL,   -- instrument currency
    gross_cents             INTEGER NOT NULL,   -- instrument currency
    withholding_cents       INTEGER NOT NULL DEFAULT 0, -- account currency
    net_cents               INTEGER NOT NULL,   -- account currency
    currency                TEXT NOT NULL,
    fx_rate_e8              INTEGER NOT NULL,
    created_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dividend_account_date ON dividend (account_id, pay_date);

CREATE TABLE IF NOT EXISTS corporate_action (
    id                    INTEGER PRIMARY KEY,
    instrument_id         INTEGER NOT NULL REFERENCES instrument(id) ON DELETE CASCADE,
    type                  TEXT NOT NULL,  -- split, reverse_split, merger, spin_off, symbol_change
    effective_date        TEXT NOT NULL,
    ratio_e8              INTEGER,        -- 2.0 -> 200000000
    cash_per_share_cents  INTEGER,
    from_symbol           TEXT,
    to_symbol             TEXT,
    processed_at          TEXT,
    note                  TEXT,
    UNIQUE (instrument_id, type, effective_date)
);

CREATE TABLE IF NOT EXISTS price_bar (
    instrument_id  INTEGER NOT NULL REFERENCES instrument(id) ON DELETE CASCADE,
    date           TEXT NOT NULL,
    open           REAL,
    high           REAL,
    low            REAL,
    close          REAL,
    adj_close      REAL,
    volume         REAL,
    source         TEXT NOT NULL DEFAULT 'yfinance',
    PRIMARY KEY (instrument_id, date)
);
CREATE INDEX IF NOT EXISTS idx_price_bar_date ON price_bar (date);

CREATE TABLE IF NOT EXISTS fx_rate (
    date       TEXT NOT NULL,
    base       TEXT NOT NULL,
    quote      TEXT NOT NULL,
    rate_e8    INTEGER NOT NULL,
    source     TEXT,
    PRIMARY KEY (date, base, quote)
);

CREATE TABLE IF NOT EXISTS watchlist (
    instrument_id  INTEGER PRIMARY KEY REFERENCES instrument(id) ON DELETE CASCADE,
    added_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS setting (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entry (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_snapshot (
    id                INTEGER PRIMARY KEY,
    snapshot_date     TEXT NOT NULL,
    account_id        INTEGER REFERENCES account(id) ON DELETE CASCADE, -- NULL = aggregate
    net_worth_cents   INTEGER NOT NULL,   -- CAD
    invested_cents    INTEGER NOT NULL,   -- CAD
    cash_cents        INTEGER NOT NULL,   -- CAD
    cost_basis_cents  INTEGER NOT NULL,   -- CAD
    day_change_cents  INTEGER NOT NULL DEFAULT 0, -- CAD
    fx_rate_e8        INTEGER NOT NULL DEFAULT 100000000,
    created_at        TEXT NOT NULL,
    UNIQUE (snapshot_date, account_id)
);

-- SQLite treats NULLs as distinct in UNIQUE constraints, so enforce a single
-- aggregate row per date explicitly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_snapshot_aggregate
    ON portfolio_snapshot (snapshot_date)
    WHERE account_id IS NULL;

CREATE TABLE IF NOT EXISTS position_snapshot (
    id                    INTEGER PRIMARY KEY,
    snapshot_id           INTEGER NOT NULL REFERENCES portfolio_snapshot(id) ON DELETE CASCADE,
    account_id            INTEGER NOT NULL REFERENCES account(id) ON DELETE CASCADE,
    instrument_id         INTEGER NOT NULL REFERENCES instrument(id) ON DELETE RESTRICT,
    shares_micros         INTEGER NOT NULL,
    price_cents           INTEGER,
    value_cents           INTEGER,   -- CAD
    cost_basis_cents      INTEGER,   -- CAD
    unrealized_pnl_cents  INTEGER,   -- CAD
    realized_pnl_cents    INTEGER NOT NULL DEFAULT 0, -- CAD
    weight_e6             INTEGER NOT NULL DEFAULT 0,
    UNIQUE (snapshot_id, account_id, instrument_id)
);
```

> **Snapshot currency:** every amount in `portfolio_snapshot` and
> `position_snapshot` is stored in **CAD**, matching the existing dashboard and
> `NetWorthSnapshot`. The ledger itself is multi-currency; conversion happens
> when the snapshot is written.

> **Note:** `txn` is used instead of `transaction` because `TRANSACTION` is a reserved SQL keyword. The Peewee model can still be named `Transaction` with `Meta.table_name = "txn"`.

### 3.3 Derived views

Create these in the migration (raw SQL) and query them from the repository:

```sql
CREATE VIEW IF NOT EXISTS v_position AS
SELECT
    l.account_id,
    l.instrument_id,
    SUM(l.remaining_quantity_micros) AS shares_micros,
    SUM(l.remaining_cost_basis_account_cents) AS cost_basis_account_cents,
    CASE
        WHEN SUM(l.remaining_quantity_micros) > 0
        THEN CAST(
            SUM(l.remaining_cost_basis_account_cents) * 1000000
            / SUM(l.remaining_quantity_micros)
            AS INTEGER
        )
        ELSE 0
    END AS avg_cost_account_cents,
    COUNT(*) AS open_lot_count
FROM lot l
WHERE l.remaining_quantity_micros > 0
GROUP BY l.account_id, l.instrument_id;

CREATE VIEW IF NOT EXISTS v_account_cash AS
SELECT
    a.id AS account_id,
    COALESCE(SUM(t.cash_delta_cents), 0) AS cash_cents
FROM account a
LEFT JOIN txn t ON t.account_id = a.id
GROUP BY a.id;

CREATE VIEW IF NOT EXISTS v_net_contributions AS
SELECT
    t.account_id,
    COALESCE(SUM(t.cash_delta_cents), 0) AS contributions_cents
FROM txn t
WHERE t.type IN ('opening', 'deposit', 'withdrawal', 'transfer_in', 'transfer_out')
GROUP BY t.account_id;
```

`v_position.avg_cost_account_cents` is the value the CLI should display as `avg_price` (account currency per share, including fees).

### 3.4 Peewee model names

| Peewee model | Table |
|---|---|
| `Account` | `account` |
| `Instrument` | `instrument` |
| `Transaction` | `txn` |
| `Lot` | `lot` |
| `LotDisposal` | `lot_disposal` |
| `Dividend` | `dividend` |
| `CorporateAction` | `corporate_action` |
| `PriceBar` | `price_bar` |
| `FxRate` | `fx_rate` |
| `Watchlist` | `watchlist` |
| `Setting` | `setting` |
| `CacheEntry` | `cache_entry` |
| `PortfolioSnapshot` | `portfolio_snapshot` |
| `PositionSnapshot` | `position_snapshot` |
| `SchemaVersion` | `schema_version` |

Retire `Holding`, `PositionLot`, `RealizedPnL`, `PositionPerformanceDaily`, `NetWorthSnapshot`, and `Recommendation` (or keep `Recommendation` with an optional `instrument_id`).

---

## 4. Accounting rules (must be implemented exactly)

### 4.1 Cash

`account cash = SUM(txn.cash_delta_cents)` over all transactions for the account.

- **Deposit:** `type='deposit'`, `cash_delta_cents=+amount` (account currency). If the deposited currency differs, convert and store `fx_rate_e8`.
- **Withdrawal:** negative.
- **Buy:** `cash_delta_cents = -(gross + fee)` in account currency.
- **Sell:** `cash_delta_cents = +(proceeds - fee)` in account currency.
- **Dividend:** `cash_delta_cents = +(net dividend)` in account currency.
- **Fee/interest:** signed cash delta.
- **Opening:** one row per account at migration or account creation, `cash_delta_cents = starting cash`. This is how legacy cash is represented; do **not** also set a column that the cash view adds.
- **Adjustment:** `cash_delta_cents = target - current` when the user runs `update-cash`. Never overwrite the balance.
- **Transfer:** `transfer_out` (negative) and `transfer_in` (positive) linked by `transfer_group_id`; net across accounts is zero.

### 4.2 Buy

Given `quantity`, `price` (instrument currency), `fee` (account currency), `fx` (account per instrument):

```
gross_instrument_cents   = round(quantity * price)
gross_account_cents      = round(gross_instrument_cents * fx)
cash_delta_cents         = -(gross_account_cents + fee_cents)
cost_basis_instrument    = gross_instrument_cents          # fees handled separately
cost_basis_account       = gross_account_cents + fee_cents
```

Create:

1. `txn(type='buy', quantity_micros, price_cents, gross_cents=gross_instrument_cents, fee_cents, currency=instrument.currency, fx_rate_e8, cash_delta_cents, external_id, trade_date)`
2. `lot(open_txn_id=txn.id, original_quantity_micros=quantity_micros, remaining_quantity_micros=quantity_micros, original_cost_basis_cents=gross_instrument_cents, remaining_cost_basis_cents=gross_instrument_cents, original_cost_basis_account_cents=cost_basis_account, remaining_cost_basis_account_cents=cost_basis_account, fx_rate_e8, fee_cents_account=fee_cents, source='buy')`

### 4.3 Sell (FIFO)

Given `quantity`, `price`, `fee`, `fx`:

```
proceeds_instrument_cents = round(quantity * price)
proceeds_account_cents    = round(proceeds_instrument_cents * fx) - fee_cents
cash_delta_cents          = proceeds_account_cents
```

Consume open lots for `(account, instrument)` ordered by `acquired_date, id`:

For each consumed quantity `q` from a lot:

```
cost_basis_account      = round(lot.remaining_cost_basis_account_cents * q / lot.remaining_quantity_micros)
cost_basis_instrument   = round(lot.remaining_cost_basis_cents * q / lot.remaining_quantity_micros)
lot.remaining_quantity_micros        -= q
lot.remaining_cost_basis_account_cents -= cost_basis_account
lot.remaining_cost_basis_cents         -= cost_basis_instrument
if remaining_quantity == 0: lot.closed_at = trade_date
create lot_disposal(...)
```

Allocate the sell fee proportionally across consumed lots for `lot_disposal.proceeds_cents`; the sum of disposal proceeds must equal `proceeds_account_cents`. Allocate the last lot the remainder to avoid rounding drift.

Realized P&L:

```
realized_pnl_account = proceeds_account - cost_basis_account
realized_pnl_instrument = proceeds_instrument - cost_basis_instrument
```

Reject the sale if total remaining quantity is insufficient. Never allow negative lots.

### 4.4 Dividends

Create one `txn(type='dividend')` and one `dividend` row:

```
gross_instrument_cents = round(shares * amount_per_share)
gross_account_cents    = round(gross_instrument_cents * fx)
net_account_cents      = gross_account_cents - withholding_cents
cash_delta_cents       = net_account_cents
```

### 4.5 Splits and corporate actions

For a split `ratio` (2:1 = 2.0):

- For every open lot of the instrument: `remaining_quantity_micros *= ratio`, `original_quantity_micros *= ratio`.
- `cost_per_share_cents /= ratio`.
- Cost basis totals do **not** change.
- Record a `corporate_action` row with `processed_at`.
- Splits do not create a cash transaction. If you need an audit trail, also write a `txn(type='split', quantity_micros=delta, cash_delta_cents=0)`.

Mergers/spin-offs are manual: use `adjustment` transactions and a note until a dedicated workflow exists.

### 4.6 All-time return

Use:

```
baseline = account.initial_cash_cents if account.initial_cash_cents != 0
           else v_net_contributions.contributions_cents
all_time_return = net_worth - baseline
```

Keep `initial_cash_cents` as a legacy/manual override so existing all-time numbers do not change after migration. New accounts can leave it at 0 and use net contributions.

### 4.7 FX

- Store the **trade-date** FX rate on every transaction and lot. Do not use today’s rate for historical cost basis.
- Cost basis in account currency uses the buy-date rate.
- Sale proceeds use the sell-date rate.
- Realized P&L in account currency therefore includes FX gain/loss, which is correct for a CAD investor holding USD assets.
- For a USD account, `fx_rate_e8 = 100_000_000`.

---

## 5. Migration strategy (v1 → v2)

### 5.1 Safety first

1. Create `data/backups/` and copy `data/portfolio.db` to `data/backups/portfolio-YYYYMMDD-HHMMSS.db` before any write.
2. Print the backup path and the migration plan.
3. Prefer building a **new database file** (`data/portfolio_v2.db`), backfilling into it, verifying it, then atomically swapping it into place. This avoids SQLite table-name collisions (`account`, `watchlist`, etc.) and lets the user keep using v1 if verification fails. Support an env override such as `STOCK_ADVISOR_DB` so the swap can be tested before it becomes the default.
4. If doing the migration in-place, create new tables with a `_v2` suffix first, then rename the old tables to `legacy_*` and the new tables to their final names inside one transaction.
5. Run the whole migration in a single transaction where possible.
6. If anything fails, restore the backup and leave the v1 database untouched.
7. Never drop legacy tables in the same release. Rename them to `legacy_*` and keep them for one release.

### 5.2 Recommended file layout

```
src/migrations/
├── __init__.py
├── runner.py          # schema_version, apply(), backup()
├── m001_initial.py    # create v2 tables + views
└── m002_backfill.py   # v1 -> v2 data backfill + reconciliation
```

`init_db()` calls `runner.apply_pending()` instead of `db.create_tables(..., safe=True)`.

### 5.3 Backfill algorithm

Assume the legacy tables are still present: `account`, `holding`, `positionlot`, `transaction`, `pricebar`, `dividend`, `realizedpnl`, `networthsnapshot`, `fxrate`, `watchlist`, `setting`, `cacheentry`.

```
1. Ensure the v2 schema exists (m001).
2. For each legacy account:
     a. create Account(name, currency=legacy.currency or infer(name),
                       initial_cash_cents=to_cents(legacy.initial_cash))
     b. create txn(type='opening', trade_date=migration_date,
                   currency=account.currency, fx_rate_e8=1e8,
                   cash_delta_cents=to_cents(legacy.cash),
                   note='Migrated opening cash balance',
                   external_id=f'legacy:account:{legacy.id}:opening')
3. For each legacy holding:
     a. instrument = get_or_create_instrument(
            symbol=holding.ticker,
            currency=infer_instrument_currency(holding.ticker, account),
            exchange=infer_exchange(holding.ticker))
     b. If legacy positionlot rows exist for (account, ticker) and are consistent
        (sum(remaining_quantity) == holding.shares):
            use them, one new Lot per legacy lot, with source='migration',
            is_estimated=0.
        Else:
            create one synthetic lot with source='migration', is_estimated=1,
            acquired_date=migration_date.
     c. For each lot, create a txn(type='buy',
            trade_date=lot.acquired_date,
            quantity_micros=..., price_cents=...,
            gross_cents=..., currency=instrument.currency,
            fx_rate_e8=..., cash_delta_cents=0,
            note='Legacy aggregate position; cash represented by opening balance',
            external_id=f'legacy:lot:{legacy_lot_id or holding.id}')
     d. Set lot.open_txn_id to that txn.
4. Copy pricebar rows:
     - map ticker -> instrument_id
     - insert into price_bar
5. Copy fxrate rows, splitting legacy pair strings into base/quote.
6. Copy networthsnapshot rows into portfolio_snapshot (account_id=NULL).
7. Copy watchlist tickers into instrument + watchlist.
8. Copy setting/cacheentry values.
9. Copy legacy transaction/dividend/realizedpnl rows into legacy_* audit tables
   (or into the new tables with cash_delta_cents=0 and note='legacy metadata').
   Do not double-count cash: the opening txn already represents current cash.
10. Reconcile (see 5.5). If reconciliation fails, abort and restore backup.
11. Rename legacy tables to legacy_*.
```

### 5.4 Currency inference

- `.TO`, `.V`, `.NE`, `.CN` → CAD (`.NE` CDRs are CAD-traded).
- `.L` → GBP, `.DE`/`.PA`/`.AS` → EUR, `.T` → JPY, otherwise USD.
- For legacy `Holding.avg_price`, assume it is in the **account currency** because `view_portfolio()` treated it that way. Mark `is_estimated=1` and `cost_currency=account.currency`. This preserves current numbers; a later `lots edit` command can correct known mistakes.
- Store `instrument.currency` separately so future live prices can be converted correctly.

### 5.5 Reconciliation queries

Run these and compare with legacy values:

```sql
-- Cash per account
SELECT a.name, v.cash_cents
FROM account a JOIN v_account_cash v ON v.account_id = a.id;

-- Shares and average cost per account/instrument
SELECT a.name, i.symbol,
       p.shares_micros, p.cost_basis_account_cents, p.avg_cost_account_cents
FROM v_position p
JOIN account a ON a.id = p.account_id
JOIN instrument i ON i.id = p.instrument_id;

-- Ledger totals
SELECT COUNT(*) FROM txn;
SELECT COUNT(*) FROM lot WHERE remaining_quantity_micros > 0;
```

Assertions:

- For every legacy account: `v_account_cash.cash_cents == to_cents(legacy.cash)`.
- For every legacy holding: `v_position.shares_micros == to_micros(legacy.shares)`.
- Weighted average cost matches `legacy.avg_price` within one cent.
- No `lot.remaining_quantity_micros < 0`.
- No `lot.remaining_quantity_micros > lot.original_quantity_micros`.
- `SUM(lot.remaining_cost_basis_account_cents)` per position equals the position’s cost basis.

### 5.6 Idempotency

- Record each migration in `schema_version` and a `migration_run` audit row with a checksum.
- Use deterministic `external_id` values for backfilled rows (`legacy:...`).
- Re-running a completed migration is a no-op.
- If a partial backfill is detected, resume from the last committed migration version; otherwise restore the backup.

### 5.7 Legacy compatibility

Keep the old tables as `legacy_*` for one release. Add a `main.py migrate-db --drop-legacy` command that deletes them only after `reconcile` passes and the user confirms.

Legacy table names produced by Peewee (use these exact names when reading v1 data):

| v1 table | v2 table | Action |
|---|---|---|
| `account` | `account` | Copy; add `currency`, `kind`, `cost_basis_method`; set `initial_cash_cents`. |
| `holding` | (removed) | Convert to `txn(type='buy')` + `lot`. |
| `positionlot` | `lot` | Use when consistent with `holding`; otherwise synthesize. |
| `transaction` | `txn` | Copy as metadata with `cash_delta_cents=0` (cash is represented by the opening txn). |
| `pricebar` | `price_bar` | Map ticker → `instrument_id`. |
| `corporateaction` | `corporate_action` | Copy; mark unprocessed if no adjustment was applied. |
| `dividend` | `dividend` | Copy if it can be linked to an instrument; otherwise keep in `legacy_dividend`. |
| `realizedpnl` | `lot_disposal` | Copy if it can be linked to a lot; otherwise keep in `legacy_realizedpnl`. |
| `positionperformancedaily` | `position_snapshot` | Optional; keep as legacy history if no snapshot link exists. |
| `networthsnapshot` | `portfolio_snapshot` | Copy as aggregate CAD rows. |
| `fxrate` | `fx_rate` | Split `pair` (e.g. `USDCAD`) into `base`/`quote`. |
| `watchlist` | `watchlist` | Map ticker → `instrument_id`. |
| `setting` | `setting` | Copy as-is. |
| `cacheentry` | `cache_entry` | Copy as-is (or let it expire and start fresh). |

> `Transaction` in v1 is the Peewee model; its table is literally `transaction`.
> The new model is also named `Transaction` but uses `Meta.table_name = "txn"`.

---

## 6. Code changes by module

### 6.1 `src/database.py`

- Replace the current models with the v2 models from section 3.4.
- Remove the duplicate `_migrate_holdings_to_lots()` definition.
- Remove `db.create_tables(..., safe=True)` from `init_db()`; call `migrations.runner.apply_pending()`.
- Enable `PRAGMA foreign_keys = ON` and `PRAGMA journal_mode = WAL` on connect.
- Keep `cache_get`/`cache_set`, `get_watchlist`/`add_to_watchlist`/`remove_from_watchlist`, but make watchlist operate on `instrument_id` and return symbols.
- Add `reconcile()` and `backup_db()` helpers.

### 6.2 `src/migrations/`

- `runner.py`: `schema_version` table, `apply_pending()`, `backup_db()`, `current_version()`.
- `m001_initial.py`: create tables/views/indexes.
- `m002_backfill.py`: the v1→v2 data migration and reconciliation.
- Each migration exposes `up(db)` and `down(db)`; log progress with `logging`.
- Use `playhouse.migrate.SqliteMigrator` for simple column changes; use `db.execute_sql()` for table rebuilds and views.

### 6.3 `src/money.py` (new)

- Helpers from section 2.3.
- Unit tests for rounding (`ROUND_HALF_UP`).

### 6.4 `src/ledger.py` (new)

The single place that writes ledger rows. All methods use `db.atomic()`.

```python
class Ledger:
    def get_or_create_instrument(symbol, currency, exchange="", name=None) -> Instrument
    def record_opening(account, amount, trade_date, note=None) -> Transaction
    def record_deposit(account, amount, currency, trade_date, fx_rate=None, note=None, external_id=None) -> Transaction
    def record_withdrawal(...) -> Transaction
    def record_buy(account, symbol, quantity, price, currency, trade_date,
                   fee=0.0, fx_rate=None, external_id=None, note=None) -> tuple[Transaction, Lot]
    def record_sell(account, symbol, quantity, price, currency, trade_date,
                    fee=0.0, fx_rate=None, method="fifo", external_id=None, note=None) -> dict
    def record_dividend(account, symbol, shares, amount_per_share, currency, pay_date,
                        withholding=0.0, fx_rate=None, external_id=None) -> tuple[Transaction, Dividend]
    def record_split(instrument, ratio, effective_date, note=None) -> CorporateAction
    def adjust_cash(account, target_amount, trade_date, note=None) -> Transaction
    def transfer(account_from, account_to, amount, trade_date, note=None) -> tuple[Transaction, Transaction]
    def get_cash(account) -> Decimal
    def get_positions(account=None, as_of=None) -> list[Position]
    def get_lots(account=None, symbol=None, open_only=True) -> list[Lot]
    def get_transactions(account=None, start=None, end=None, symbol=None) -> list[Transaction]
    def get_realized_pnl(account=None, start=None, end=None, symbol=None) -> list[LotDisposal]
    def reconcile(account=None) -> list[dict]
```

`record_sell` returns a summary dict:

```python
{
    "proceeds": Decimal,
    "cost_basis": Decimal,
    "realized_pnl": Decimal,
    "currency": "USD",
    "disposals": [...],
}
```

### 6.5 `src/portfolio.py` — compatibility facade

Keep the public API so `main.py`, `brief.py`, `newsletter.py`, `chat_advisor.py` and tests do not all need to change at once.

- `load()` returns the existing dict shape, computed from `ledger.get_positions()` and `ledger.get_cash()`:
  ```python
  {
      "accounts": {
          "USD": {
              "holdings": {
                  "AAPL": {
                      "shares": 10.0,
                      "avg_price": 150.50,      # account currency, incl. fees
                      "currency": "USD",        # instrument currency
                      "cost_basis": 1505.0,     # account currency
                      "lot_count": 2,
                  }
              },
              "cash": 3495.0,
              "initial_cash": 5000.0,
              "currency": "USD",
          }
      }
  }
  ```
- `snapshot()` returns the same keys as today, plus `currency` and `price_account` per row. Keep all values in CAD for the aggregate.
- `add_position(...)` → `ledger.record_buy(...)`.
- `sell_position(...)` → `ledger.record_sell(...)`. Return `(proceeds_account, fx_rate)` to preserve the CLI contract.
- `deposit_cash(...)` → `ledger.record_deposit(...)`.
- `update_cash(...)` → `ledger.adjust_cash(...)` (absolute target → adjustment transaction).
- `set_initial_cash(...)` → set `account.initial_cash_cents` (no cash change).
- `remove_position(...)` → for real sales, use `record_sell`; for corrections, write an `adjustment` write-off with a note. Add a `--price` option to the CLI so it can record a real sale when a price is supplied.
- `get_account_holdings(...)` → derive from `v_position`.
- `log_net_worth()` → compute from the ledger and write `portfolio_snapshot` + `position_snapshot`.
- `get_history()` → read `portfolio_snapshot` (aggregate rows).
- Delete `save()` and `ensure_account_exists()` after updating all callers; if kept temporarily, raise `DeprecationWarning` and never touch holdings.

**Critical FX fix:** `main.py::view_portfolio()` and `_build_holding_row()` must convert each holding’s live price from `holding["currency"]` to the account currency (and then to CAD for the grand total), not assume the account currency equals the instrument currency. Add a `price_account` value to each row and use it for cost/value/day-change math.

### 6.6 `src/repositories/sqlite_repo.py` and `src/services/portfolio_service.py`

- Replace `Holding` queries with `v_position`/`Lot` queries.
- Add repository methods:
  - `get_transactions(account, start, end, symbol)`
  - `get_lots(account, symbol, open_only)`
  - `get_realized_pnl(account, start, end)`
  - `get_dividends(account, start, end)`
  - `get_cost_basis(account, symbol)`
- `get_price_bars`/`save_price_bars` map ticker → `instrument_id`.
- `PortfolioService.snapshot()` can keep delegating to `portfolio.snapshot()`.

### 6.7 `main.py`

Existing command UX stays the same. Add:

| Command | Purpose |
|---|---|
| `history [--account] [--ticker] [--from] [--to]` | List ledger transactions (date, type, symbol, qty, price, fee, cash delta). |
| `lots [--account] [--ticker] [--all]` | Show open (or all) tax lots with acquired date, remaining qty, cost/share, cost basis. |
| `realized [--account] [--ticker] [--year]` | Realized P&L per disposal and totals. |
| `dividend add TICKER SHARES AMOUNT_PER_SHARE --account --pay-date [--withholding]` | Record an actual dividend. |
| `dividend list [--account] [--year]` | List recorded dividends. |
| `transfer AMOUNT --from USD --to CAD [--date]` | Move cash between accounts with linked transactions. |
| `reconcile [--account]` | Verify cash/positions/cost basis against the ledger and report discrepancies. |
| `migrate-db [--drop-legacy]` | Run/verify the v1→v2 migration and optionally drop legacy tables. |
| `import-csv PATH` | Import broker transactions (CSV) idempotently via `external_id`. |

Update existing commands:

- `add-stock`: add `--date`, `--fee`, `--fx-rate`; record a buy lot.
- `sell-stock`: add `--date`, `--fee`, `--fx-rate`; print realized P&L and remaining lots.
  **Behavior change:** proceeds stay in the **selling account**. The old code
  silently credited the CAD account; that is an accounting bug. If the user
  wants CAD, record an explicit `transfer` (two linked transactions) and print
  both legs. Do not credit a different account inside `record_sell`.
- `deposit`: add `--date`; record a deposit transaction.
- `update-cash`: keep the absolute target, but record an adjustment.
- `set-initial`: set the baseline only; do not touch cash.
- `remove-stock`: add `--price`; with a price, record a sale; without, write off with a warning and require confirmation.
- `export`: add `--transactions` / `--lots` to export history.
- `dividends`: keep the projected-income view; point users to `dividend add`/`dividend list` for actuals.

### 6.8 `src/brief.py`

- Replace `Account.select()` + `acc.holdings` with `ledger.get_positions()` or `portfolio.load()`.
- `_upsert_portfolio_holdings()` should write positions derived from `v_position`, including `currency` and `cost_basis_cad`.
- `portfolio_accounts.cash` should come from `v_account_cash`, not `Account.cash`.
- Add an optional `_upsert_ledger()` that mirrors `txn`, `lot`, `lot_disposal` and `dividend` to Supabase with deterministic IDs (`account:external_id` or `account:type:trade_date:sequence`).
- Keep the existing remote fallback logic; it remains useful for a fresh CI runner.

### 6.9 `src/newsletter.py`, `src/chat_advisor.py`, `src/advisor.py`

- They already consume `portfolio.load()` / `portfolio.snapshot()`; keep them on the facade.
- `chat_advisor.py` currently imports `Account` directly — replace that with `portfolio.load()` / `ledger.get_positions()`.

### 6.10 `src/setup.py`

- Stop creating `portfolio.json` and `settings.json`.
- Call `init_db()` and write default settings to the `setting` table.
- Keep the JSON migration code in `database.py` for one release, then remove it.

### 6.11 `dashboard.py` and `sql/supabase_brief.sql`

- Keep the existing `portfolio_snapshots` / `portfolio_snapshot_items` writes so the dashboard keeps working.
- Add Supabase mirror tables (section 7) and a “Transactions” / “Lots” tab in the dashboard.
- Update `portfolio_holdings` to include `currency` and `cost_basis_cad`.
- Add RLS policies for the new tables.

### 6.12 Tests

Update fixtures to create/drop the new tables. Add tests for:

- `to_cents`/`from_cents`/`to_micros` rounding.
- Buy creates one txn + one lot; cash decreases correctly.
- Two buys create two lots; weighted average cost is correct.
- Partial FIFO sell consumes the oldest lot first.
- Sell across multiple lots; realized P&L sum matches.
- Sell more than held raises and changes nothing.
- Fees are included in cost basis/proceeds.
- USD stock in a CAD account: cost basis uses the buy FX; sale uses the sell FX; realized P&L includes FX.
- Dividend gross/net/withholding and cash.
- 2:1 split doubles shares and halves cost/share; cost basis unchanged.
- Deposit/withdrawal/update-cash/transfer create the right transactions and reconcile.
- `portfolio.load()` returns the legacy dict shape and correct values.
- `portfolio.snapshot()` totals in CAD with per-instrument currency conversion.
- Migration from a v1 DB: counts, cash, shares, avg cost, idempotency, and backup-on-failure.
- `reconcile()` returns no discrepancies after randomized buy/sell/dividend sequences.

---

## 7. Supabase mirror schema

Append to `sql/supabase_brief.sql` (keep the existing tables):

```sql
create table if not exists portfolio_transactions (
    id                 bigserial primary key,
    account            text not null,
    external_id        text,
    type               text not null,
    trade_date         date not null,
    symbol             text,
    instrument_currency text,
    quantity           numeric(24,8) not null default 0,
    price              numeric(18,4) not null default 0,
    fee                numeric(18,2) not null default 0,
    currency           text not null,
    fx_rate            numeric(18,8) not null default 1,
    cash_delta         numeric(18,2) not null default 0,
    note               text,
    created_at         timestamptz not null default now(),
    unique (account, external_id)
);
create index if not exists idx_portfolio_txn_account_date
    on portfolio_transactions (account, trade_date);

create table if not exists portfolio_lots (
    id                  bigserial primary key,
    account             text not null,
    symbol              text not null,
    acquired_date       date not null,
    original_quantity   numeric(24,8) not null,
    remaining_quantity  numeric(24,8) not null,
    cost_per_share      numeric(18,4) not null,
    cost_currency       text not null,
    cost_basis          numeric(18,2) not null,
    fx_rate             numeric(18,8) not null default 1,
    source              text,
    is_estimated        boolean not null default false,
    closed_at           date
);
create index if not exists idx_portfolio_lots_position
    on portfolio_lots (account, symbol, remaining_quantity);

create table if not exists portfolio_realized_pnl (
    id             bigserial primary key,
    account        text not null,
    symbol         text not null,
    lot_id         bigint,
    sell_date      date not null,
    quantity       numeric(24,8) not null,
    proceeds       numeric(18,2) not null,
    cost_basis     numeric(18,2) not null,
    realized_pnl   numeric(18,2) not null,
    currency       text not null,
    fx_rate        numeric(18,8) not null default 1
);
create index if not exists idx_portfolio_realized_account_date
    on portfolio_realized_pnl (account, sell_date);

create table if not exists portfolio_dividends (
    id                 bigserial primary key,
    account            text not null,
    symbol             text not null,
    ex_date            date,
    pay_date           date not null,
    shares             numeric(24,8) not null,
    amount_per_share   numeric(18,4) not null,
    gross              numeric(18,2) not null,
    withholding        numeric(18,2) not null default 0,
    net                numeric(18,2) not null,
    currency           text not null,
    fx_rate            numeric(18,8) not null default 1
);

alter table portfolio_transactions enable row level security;
alter table portfolio_lots enable row level security;
alter table portfolio_realized_pnl enable row level security;
alter table portfolio_dividends enable row level security;

create policy "allow read" on portfolio_transactions for select using (true);
create policy "allow read" on portfolio_lots for select using (true);
create policy "allow read" on portfolio_realized_pnl for select using (true);
create policy "allow read" on portfolio_dividends for select using (true);

create policy "allow write" on portfolio_transactions for insert with check (true);
create policy "allow update" on portfolio_transactions for update using (true) with check (true);
create policy "allow delete" on portfolio_transactions for delete using (true);
-- Repeat the write/update/delete policies for portfolio_lots,
-- portfolio_realized_pnl and portfolio_dividends.
```

Mirroring rules:

- Use deterministic `external_id` values so re-running `brief --persist` is idempotent.
- Push transactions/lots incrementally, not just the current snapshot.
- Keep `portfolio_holdings` and `portfolio_accounts` for the newsletter and CI fallback.
- Never let a mirror failure fail the brief run; log and continue.

---

## 8. Implementation order and commit plan

Each step must leave `ruff check .` and `pytest` green. Commit after each step.

1. **`feat(db): add versioned migration runner`**
   - `schema_version`, `src/migrations/runner.py`, backup helper, `init_db()` wiring.
2. **`feat(db): add ledger, instrument, lot and disposal models`**
   - v2 models + `m001_initial.py` tables/views/indexes.
3. **`feat(money): add exact money and quantity helpers`**
   - `src/money.py` + tests.
4. **`feat(ledger): implement buys, sells, dividends and cash operations`**
   - `src/ledger.py` with FIFO, fees, FX, dividends, splits, adjustments, transfers.
5. **`refactor(portfolio): make portfolio API a ledger facade`**
   - `load`, `snapshot`, `add_position`, `sell_position`, `deposit_cash`, `update_cash`, `set_initial_cash`, `remove_position`, `log_net_worth`.
6. **`fix(portfolio): convert per-instrument currency correctly`**
   - Update `view_portfolio`/`_build_holding_row`/`snapshot` to use `holding["currency"]`.
7. **`feat(db): backfill legacy holdings into opening lots`**
   - `m002_backfill.py`, reconciliation, backup, idempotency.
8. **`feat(cli): add history, lots, realized, dividends and reconcile commands`**
   - New commands and `--date`/`--fee`/`--fx-rate` options.
9. **`refactor(brief): read positions from the ledger`**
   - Replace direct `Account`/`Holding` access; update snapshot mirroring.
10. **`feat(supabase): add ledger mirror tables and RLS`**
    - SQL + persist logic + dashboard tab.
11. **`test(db): cover FIFO, FX, dividends, splits and migration`**
    - Full test suite and randomized reconciliation.
12. **`docs: document the ledger schema and migration`**
    - README architecture update and a short “how to migrate” section.

Do **not** combine steps 1–4 with the backfill; the migration must be reviewable on its own.

---

## 9. Acceptance criteria

- [ ] `python main.py add-stock AAPL 10 185.50 --account USD --date 2026-03-14 --fee 4.95` writes one `txn` and one `lot` with the date, quantity, price, fee and FX rate.
- [ ] `python main.py history --ticker AAPL` shows the individual purchase with its date.
- [ ] `python main.py lots --ticker AAPL` shows the remaining lot(s), acquired date, cost/share and cost basis.
- [ ] `python main.py sell-stock AAPL 4 210.10 --account USD --fee 4.95` consumes FIFO lots and prints realized P&L.
- [ ] `python main.py realized --ticker AAPL` shows the disposal and total realized gain.
- [ ] `python main.py view-portfolio` produces the same layout as before, with correct FX for US stocks in CAD accounts.
- [ ] `python main.py deposit 1000 --account CAD --date 2026-01-05` creates a deposit transaction; cash is derived and reconciles.
- [ ] `python main.py update-cash 2500 --account CAD` creates an adjustment transaction, not a silent overwrite.
- [ ] `python main.py dividend add AAPL 10 0.25 --account USD --pay-date 2026-05-15` records a dividend and credits cash.
- [ ] A 2:1 split doubles shares and halves cost/share without changing total cost basis.
- [ ] `python main.py reconcile` reports zero discrepancies.
- [ ] Migrating an existing `data/portfolio.db` preserves cash, shares and average cost; re-running the migration is a no-op.
- [ ] A failed migration restores the backup and leaves the v1 DB usable.
- [ ] `brief --persist` still writes the snapshot tables and additionally mirrors transactions/lots.
- [ ] The Streamlit dashboard still loads the existing tabs; a Transactions/Lots tab reads the new mirror.
- [ ] `ruff check .` and `pytest` pass.
- [ ] No legacy JSON files are created on a fresh install.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Legacy `avg_price` currency is ambiguous | Assume account currency (matches current UI), mark `is_estimated=1`, provide `lots edit` to correct. |
| Float drift in old data | Convert to cents once at migration; store exact integers thereafter. |
| SQLite `ALTER TABLE` limitations | Use `playhouse.migrate` for simple changes; rebuild tables with `CREATE TABLE new`, `INSERT INTO new SELECT`, `DROP`, `RENAME`. |
| Long migration / user data loss | Backup first, run in a transaction, print progress, abort on reconciliation failure. |
| Behavior changes break the CLI | Keep `portfolio.load()`/`snapshot()` shapes stable; update output only where FX was wrong. |
| Supabase publishable key cannot write | Re-run the RLS section of `sql/supabase_brief.sql`; test with a dry-run insert. |
| Users have positions with no known purchase date | Synthetic lot with `acquired_date=migration_date`, `is_estimated=1`, note “legacy aggregate”; allow later edits. |
| Split/merger edge cases | Implement splits only; treat mergers/spin-offs as manual adjustments with notes until a dedicated workflow exists. |
| Fees in different currencies | Define `fee_cents` as account currency; convert at entry; document in the CLI help. |

---

## Appendix A — Worked examples

### A.1 Buy and partial FIFO sell (USD account)

Buy 10 AAPL at $150, fee $5:

```
gross = 10 * 150 = 1500 USD
cash_delta = -(1500 + 5) = -1505 USD
lot: qty 10, cost basis 1500 USD, account cost basis 1505 USD
avg_price shown = 1505 / 10 = 150.50 USD
```

Sell 4 at $200, fee $4:

```
proceeds = 4 * 200 - 4 = 796 USD
cost basis = 4/10 * 1505 = 602 USD
realized = 796 - 602 = 194 USD
remaining lot: 6 shares, account cost basis 903 USD
avg_price shown = 903 / 6 = 150.50 USD
```

### A.2 USD stock in a CAD account (FX)

Buy 10 shares at $100 USD, buy-date FX 1.35, fee CAD 5:

```
gross_instrument = 1000 USD
gross_account = 1000 * 1.35 = 1350 CAD
cash_delta = -(1350 + 5) = -1355 CAD
lot account cost basis = 1355 CAD
avg_price shown = 135.50 CAD/share
```

Sell all 10 at $110 USD, sell-date FX 1.30, fee CAD 5:

```
proceeds_instrument = 1100 USD
proceeds_account = 1100 * 1.30 - 5 = 1425 CAD
cost basis = 1355 CAD
realized = 1425 - 1355 = 70 CAD
```

The 70 CAD gain includes a 50 CAD FX loss versus the 100 CAD price gain; the ledger captures both.

### A.3 2:1 split

Before: 10 shares, cost/share 150.00, cost basis 1500.
After: 20 shares, cost/share 75.00, cost basis 1500. No cash transaction.

---

## Appendix B — Useful queries

```sql
-- Current positions with account and instrument details
SELECT a.name AS account, i.symbol, i.currency,
       p.shares_micros / 1000000.0 AS shares,
       p.avg_cost_account_cents / 100.0 AS avg_cost,
       p.cost_basis_account_cents / 100.0 AS cost_basis
FROM v_position p
JOIN account a ON a.id = p.account_id
JOIN instrument i ON i.id = p.instrument_id
ORDER BY a.name, i.symbol;

-- Cash per account
SELECT a.name, v.cash_cents / 100.0 AS cash
FROM account a JOIN v_account_cash v ON v.account_id = a.id;

-- Transaction history for one ticker
SELECT t.trade_date, t.type, t.quantity_micros / 1000000.0 AS qty,
       t.price_cents / 100.0 AS price, t.fee_cents / 100.0 AS fee,
       t.cash_delta_cents / 100.0 AS cash_delta, t.note
FROM txn t
JOIN instrument i ON i.id = t.instrument_id
WHERE i.symbol = 'AAPL'
ORDER BY t.trade_date, t.id;

-- Realized P&L by year
SELECT substr(d.realized_at, 1, 4) AS year,
       SUM(d.realized_pnl_cents) / 100.0 AS realized_pnl
FROM lot_disposal d
GROUP BY year
ORDER BY year;

-- Open lots for one ticker
SELECT l.acquired_date, l.remaining_quantity_micros / 1000000.0 AS qty,
       l.cost_per_share_cents / 100.0 AS cost_per_share,
       l.remaining_cost_basis_account_cents / 100.0 AS cost_basis
FROM lot l
JOIN instrument i ON i.id = l.instrument_id
WHERE i.symbol = 'AAPL' AND l.remaining_quantity_micros > 0
ORDER BY l.acquired_date, l.id;
```

---

## Appendix C — Definition of done checklist for the agent

- [ ] All new tables/views exist and are created by versioned migrations.
- [ ] No code path writes `Holding` or mutates `Account.cash` directly.
- [ ] Every buy/sell/dividend/cash change goes through `src/ledger.py` inside `db.atomic()`.
- [ ] FIFO lot consumption is covered by tests, including multi-lot sells and FX.
- [ ] `portfolio.load()` and `portfolio.snapshot()` keep their existing shapes.
- [ ] `view-portfolio` uses per-instrument currency conversion.
- [ ] The v1→v2 migration is idempotent, backed up, and reconciled.
- [ ] New CLI commands are documented in `README.md`.
- [ ] Supabase mirror tables and RLS policies are in `sql/supabase_brief.sql`.
- [ ] The dashboard still works and optionally shows transactions/lots.
- [ ] `ruff check .` and `pytest` pass.
- [ ] Commits follow the plan in section 8.
