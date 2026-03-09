"""
Forecasto API client.

Wraps all four API endpoints behind a thin abstraction:
  - fetch_sales(start_date, end_date) -> pd.DataFrame
  - fetch_stock(date) -> pd.DataFrame
  - fetch_products() -> pd.DataFrame
  - fetch_losses(date) -> pd.DataFrame

All dates are passed as Python date objects; the client formats them for the API.
Returns DataFrames with English column names (see schemas.py for the mapping).

Retry and timeout behaviour is configurable via ApiConfig.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.api.schemas import (
    parse_loss_record,
    parse_product_record,
    parse_sales_record,
    parse_stock_record,
)
from src.utils.config import ApiConfig, get_api_token
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DATE_FMT = "%d.%m.%Y"  # Forecasto API date format: dd.MM.yyyy


class ForecastoAPIError(Exception):
    """Raised when an API call fails after all retries."""


class ForecastoClient:
    """
    Thin HTTP client for the Forecasto platform APIs.

    Usage:
        client = ForecastoClient.from_config(api_config)
        sales_df = client.fetch_sales(date(2026, 1, 1), date(2026, 3, 1))
    """

    def __init__(self, base_url: str, token: str, timeout: int, retries: int, backoff_factor: float):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._session = self._build_session(retries, backoff_factor)

    @classmethod
    def from_config(cls, cfg: ApiConfig) -> "ForecastoClient":
        token = get_api_token()
        return cls(
            base_url=cfg.base_url,
            token=token,
            timeout=cfg.timeout,
            retries=cfg.retries,
            backoff_factor=cfg.backoff_factor,
        )

    def _build_session(self, retries: int, backoff_factor: float) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _post(self, endpoint: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        logger.debug("POST %s payload=%s", url, list(payload.keys()))
        t0 = time.perf_counter()
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            elapsed = time.perf_counter() - t0
            logger.debug("Response %s in %.2fs", resp.status_code, elapsed)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise ForecastoAPIError(f"Timeout after {self._timeout}s calling {url}")
        except requests.exceptions.ConnectionError as exc:
            raise ForecastoAPIError(f"Connection error calling {url}: {exc}")
        except requests.exceptions.HTTPError as exc:
            raise ForecastoAPIError(f"HTTP {exc.response.status_code} calling {url}: {exc.response.text[:200]}")

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_sales(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch daily sales records for a date range.

        Returns a DataFrame with English column names.
        """
        logger.info("Fetching sales %s → %s", start_date, end_date)
        payload = {
            "token": self._token,
            "START_DATE": start_date.strftime(_DATE_FMT),
            "FINISH_DATE": end_date.strftime(_DATE_FMT),
        }
        records = self._post("sales", payload)
        if not isinstance(records, list):
            raise ForecastoAPIError(f"Unexpected sales response type: {type(records)}")
        rows = [parse_sales_record(r) for r in records]
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("Sales API returned 0 records for %s → %s", start_date, end_date)
            return pd.DataFrame(columns=list(parse_sales_record({}).keys()))
        df["date"] = pd.to_datetime(df["date"], format=_DATE_FMT, errors="coerce")
        df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0)
        df["sales_amount"] = pd.to_numeric(df["sales_amount"], errors="coerce").fillna(0)
        logger.info("Sales: %d records fetched", len(df))
        return df

    def fetch_stock(self, target_date: date) -> pd.DataFrame:
        """
        Fetch stock balances for a single date.

        Returns a DataFrame with English column names.
        """
        logger.info("Fetching stock for %s", target_date)
        payload = {
            "token": self._token,
            "Date": target_date.strftime(_DATE_FMT),
        }
        records = self._post("inventory/stock", payload)
        if not isinstance(records, list):
            raise ForecastoAPIError(f"Unexpected stock response type: {type(records)}")
        rows = [parse_stock_record(r) for r in records]
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("Stock API returned 0 records for %s", target_date)
            return pd.DataFrame(columns=list(parse_stock_record({}).keys()))
        df["date"] = pd.to_datetime(df["date"], format=_DATE_FMT, errors="coerce")
        df["stock_balance"] = pd.to_numeric(df["stock_balance"], errors="coerce").fillna(0)
        logger.info("Stock: %d records fetched", len(df))
        return df

    def fetch_products(self) -> pd.DataFrame:
        """
        Fetch product metadata (expiration days, shipment multiple, UoM, etc.).

        Returns a DataFrame with English column names.
        """
        logger.info("Fetching product info")
        payload = {"token": self._token}
        response = self._post("backend/delivery_info/api/v1/GetAll", payload)
        if not isinstance(response, dict) or not response.get("success"):
            raise ForecastoAPIError(f"Product info API returned unexpected response: {response}")
        items = response.get("items", [])
        rows = [parse_product_record(item) for item in items]
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("Product info API returned 0 items")
        else:
            df["expiration_days"] = pd.to_numeric(df["expiration_days"], errors="coerce")
            df["shipment_multiple"] = pd.to_numeric(df["shipment_multiple"], errors="coerce").fillna(1)
            logger.info("Products: %d records fetched", len(df))
        return df

    def fetch_losses(self, target_date: date) -> pd.DataFrame:
        """
        Fetch write-off / loss records for a single date.

        Returns a DataFrame with English column names.
        """
        logger.info("Fetching losses for %s", target_date)
        payload = {
            "token": self._token,
            "Date": target_date.strftime(_DATE_FMT),
        }
        records = self._post("loss/getall", payload)
        if not isinstance(records, list):
            raise ForecastoAPIError(f"Unexpected losses response type: {type(records)}")
        rows = [parse_loss_record(r) for r in records]
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("Loss API returned 0 records for %s", target_date)
            return pd.DataFrame(columns=list(parse_loss_record({}).keys()))
        df["date"] = pd.to_datetime(df["date"], format=_DATE_FMT, errors="coerce")
        df["loss_qty"] = pd.to_numeric(df["loss_qty"], errors="coerce").fillna(0)
        df["loss_amount"] = pd.to_numeric(df["loss_amount"], errors="coerce").fillna(0)
        logger.info("Losses: %d records fetched", len(df))
        return df
