"""
Data loader: abstract over API vs local sample source.

Usage:
    loader = DataLoader(config)
    sales_df   = loader.load_sales(start_date, end_date)
    stock_df   = loader.load_stock(date)
    products_df = loader.load_products()
    losses_df  = loader.load_losses(date)

In "local" mode, files are read from config.data.samples_dir.
In "api" mode, data is fetched from the live Forecasto API and
cached to config.data.raw_dir as parquet for reproducibility.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Source-agnostic data loader.

    All public methods return DataFrames with the internal English schema
    (see src/data/schema.py and src/api/schemas.py for column names).
    """

    def __init__(self, config: Config):
        self._cfg = config
        self._source = config.data.source
        self._raw_dir = Path(config.data.raw_dir)
        self._samples_dir = Path(config.data.samples_dir)
        self._skip_if_exists = config.data.skip_if_exists
        self._client = None  # created lazily on first live API call

        self._raw_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """
        Return the API client, initializing it on first use.

        Kept lazy so commands like build-dataset, train, predict that only
        read cached parquet files never require FORECASTO_TOKEN to be set.
        The token is only validated when an actual live request is about to go out.
        """
        if self._client is None:
            from src.api.client import ForecastoClient
            self._client = ForecastoClient.from_config(self._cfg.api)
        return self._client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_sales(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load sales data.

        Local mode: reads data/samples/sales.parquet (date range ignored).
        API mode with dates: fetches from API and caches to data/raw/.
        API mode without dates: merges all cached sales_*.parquet files in
          data/raw/ — useful after fetch-data has already been run.
        """
        if self._source == "local":
            return self._read_local("sales.parquet")

        # API mode — specific date range requested
        if start_date is not None and end_date is not None:
            cache_path = self._raw_dir / f"sales_{start_date}_{end_date}.parquet"
            if self._skip_if_exists and cache_path.exists():
                logger.info("Loading cached sales from %s", cache_path)
                return pd.read_parquet(cache_path)
            df = self._get_client().fetch_sales(start_date, end_date)
            df.to_parquet(cache_path, index=False)
            logger.info("Saved sales to %s", cache_path)
            return df

        # API mode — no dates given: load from all cached sales files
        return self._load_all_cached(prefix="sales_", label="sales")

    def load_stock(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """
        Load stock balance snapshots.

        Local mode: reads data/samples/stock.parquet.
        API mode with date: fetches that specific date and caches it.
        API mode without date: merges all cached stock_*.parquet files.
        """
        if self._source == "local":
            return self._read_local("stock.parquet")

        if target_date is not None:
            cache_path = self._raw_dir / f"stock_{target_date}.parquet"
            if self._skip_if_exists and cache_path.exists():
                logger.debug("Loading cached stock from %s", cache_path)
                return pd.read_parquet(cache_path)
            df = self._get_client().fetch_stock(target_date)
            df.to_parquet(cache_path, index=False)
            return df

        return self._load_all_cached(prefix="stock_", label="stock")

    def load_products(self) -> pd.DataFrame:
        """Load product metadata (static; no date parameter)."""
        if self._source == "local":
            return self._read_local("products.parquet")
        cache_path = self._raw_dir / "products.parquet"
        if self._skip_if_exists and cache_path.exists():
            logger.info("Loading cached products from %s", cache_path)
            return pd.read_parquet(cache_path)
        df = self._get_client().fetch_products()
        df.to_parquet(cache_path, index=False)
        return df

    def load_losses(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """
        Load write-off / loss records.

        Local mode: reads data/samples/losses.parquet.
        API mode with date: fetches that specific date and caches it.
        API mode without date: merges all cached losses_*.parquet files.
        """
        if self._source == "local":
            return self._read_local("losses.parquet")

        if target_date is not None:
            cache_path = self._raw_dir / f"losses_{target_date}.parquet"
            if self._skip_if_exists and cache_path.exists():
                logger.debug("Loading cached losses from %s", cache_path)
                return pd.read_parquet(cache_path)
            df = self._get_client().fetch_losses(target_date)
            df.to_parquet(cache_path, index=False)
            return df

        return self._load_all_cached(prefix="losses_", label="losses")

    def fetch_all_api(self, start_date: date, end_date: date) -> None:
        """
        Fetch all four data sources via API for a date range and cache to raw_dir.
        Used by the fetch-data CLI command.

        Stock and loss endpoints are per-day only (API limitation), so this
        makes 2 × N_days calls for those two sources.

        Failure handling (dead-letter pattern):
        - Each day's result is written to disk immediately after a successful fetch.
        - If a day fails (network error, timeout, bad response), the error is
          recorded in data/raw/fetch_errors.log and the loop continues.
        - Failed days are NOT written to disk, so re-running the command will
          automatically retry only the failed days (already-cached days are skipped).
        - At the end, a summary shows exactly which days failed so you know
          what to re-run or investigate.
        """
        if self._source != "api":
            logger.warning("fetch_all_api called in local mode — no API calls made")
            return

        total_days = (end_date - start_date).days + 1
        error_log_path = self._raw_dir / "fetch_errors.log"

        print(f"\nFetching data from API: {start_date} → {end_date} ({total_days} days)")
        print("=" * 60)

        # --- Sales (one call for the full range) ---
        print("[1/3] Sales data (single range call)...", end=" ", flush=True)
        try:
            self.load_sales(start_date, end_date)
            print("done.")
        except Exception as exc:
            print(f"FAILED: {exc}")
            self._log_error(error_log_path, "sales", str(start_date), str(end_date), exc)
            print(f"       Error logged to {error_log_path}")
            print("       Cannot continue without sales data. Fix the error and re-run.")
            return

        # --- Product metadata (one call, no date) ---
        print("[2/3] Product metadata (single call)...", end=" ", flush=True)
        try:
            self.load_products()
            print("done.")
        except Exception as exc:
            print(f"FAILED: {exc}")
            self._log_error(error_log_path, "products", "-", "-", exc)
            print(f"       Error logged to {error_log_path}")
            print("       Cannot continue without product metadata. Fix the error and re-run.")
            return

        # --- Stock + Losses (one call per day each) ---
        print(f"[3/3] Stock + Losses (one call per day × {total_days} days)...")
        print(f"      Failures are logged and skipped — re-run to retry failed days.\n")

        d = start_date
        day_num = 0
        skipped = 0
        fetched = 0
        failed_days: list[tuple[date, str]] = []  # (date, reason)

        while d <= end_date:
            day_num += 1

            stock_cached = (self._raw_dir / f"stock_{d}.parquet").exists()
            losses_cached = (self._raw_dir / f"losses_{d}.parquet").exists()

            if self._skip_if_exists and stock_cached and losses_cached:
                skipped += 1
            else:
                day_ok = True

                if not (self._skip_if_exists and stock_cached):
                    try:
                        self.load_stock(d)
                    except Exception as exc:
                        reason = f"stock: {exc}"
                        self._log_error(error_log_path, "stock", str(d), "-", exc)
                        failed_days.append((d, reason))
                        day_ok = False

                if not (self._skip_if_exists and losses_cached):
                    try:
                        self.load_losses(d)
                    except Exception as exc:
                        reason = f"losses: {exc}"
                        self._log_error(error_log_path, "losses", str(d), "-", exc)
                        # Only append if not already recorded for this day
                        if not failed_days or failed_days[-1][0] != d:
                            failed_days.append((d, reason))
                        day_ok = False

                if day_ok:
                    fetched += 1

            # Progress line every 10 days or on the last day
            if day_num % 10 == 0 or d == end_date:
                pct = int(100 * day_num / total_days)
                bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                fail_note = f", {len(failed_days)} failed" if failed_days else ""
                print(
                    f"  [{bar}] {pct:3d}%  day {day_num}/{total_days}  "
                    f"({fetched} fetched, {skipped} cached{fail_note})",
                    flush=True,
                )

            d += timedelta(days=1)

        # --- Final summary ---
        print(f"\n{'=' * 60}")
        print(f"Fetch complete.")
        print(f"  Fetched from API : {fetched} days")
        print(f"  Loaded from cache: {skipped} days")
        print(f"  Failed           : {len(failed_days)} days")

        if failed_days:
            print(f"\n  Failed days (will be retried automatically on next run):")
            for fail_date, reason in failed_days:
                print(f"    {fail_date}  —  {reason}")
            print(f"\n  Full error details: {error_log_path}")
            print(f"  Re-run the same fetch-data command to retry only the failed days.")
        else:
            # Clean up error log if everything succeeded
            if error_log_path.exists():
                error_log_path.unlink()

        print(f"\nRaw files saved to: {self._raw_dir.resolve()}")
        print("Next step: python -m src.cli.main build-dataset\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_cached(self, prefix: str, label: str) -> pd.DataFrame:
        """
        Find all parquet files in raw_dir whose name starts with `prefix`,
        concatenate them, and return a single DataFrame.

        Used by load_sales / load_stock / load_losses when no date is specified
        in API mode — i.e. after fetch-data has already populated the cache.
        """
        files = sorted(self._raw_dir.glob(f"{prefix}*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No cached {label} files found in {self._raw_dir} "
                f"(looked for {prefix}*.parquet).\n"
                f"Run fetch-data first:  python -m src.cli.main fetch-data "
                f"--start-date YYYY-MM-DD --end-date YYYY-MM-DD"
            )
        parts = [pd.read_parquet(f) for f in files]
        df = pd.concat(parts, ignore_index=True)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        logger.info(
            "Loaded %s from %d cached files → %d rows (date range: %s → %s)",
            label, len(files), len(df),
            df["date"].min().date() if "date" in df.columns else "?",
            df["date"].max().date() if "date" in df.columns else "?",
        )
        return df

    def _log_error(
        self,
        log_path: Path,
        endpoint: str,
        date_from: str,
        date_to: str,
        exc: Exception,
    ) -> None:
        """Append a failure record to the error log file."""
        import traceback
        from datetime import datetime as dt

        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] endpoint={endpoint} "
            f"date_from={date_from} date_to={date_to}\n"
            f"  Error: {exc}\n"
            f"  Traceback: {traceback.format_exc().strip()}\n"
            f"{'-' * 60}\n"
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
        logger.debug("Error logged to %s", log_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_local(self, filename: str) -> pd.DataFrame:
        path = self._samples_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Sample file not found: {path}\n"
                f"Run: python data/samples/generate_samples.py"
            )
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        logger.debug("Loaded %d rows from %s", len(df), path)
        return df
