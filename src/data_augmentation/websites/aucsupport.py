"""AucSupport (aucsupport.com) 爬虫模块

オークションサポートサイトから中古車データを取得する。
"""

import logging
import pickle
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

REQUEST_INTERVAL_SEC = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://www.aucsupport.com/",
    "Connection": "close",
}

EXPECTED_HEADERS = [
    "メーカー",
    "車種",
    "年式",
    "シフト",
    "走行",
    "型式",
    "評価",
    "価格(万円)",
    "更新年月",
    "グレード",
    "排気量",
    "車検",
    "色",
    "装備",
]


def clean_text(s: str | None) -> str:
    """テキストの正規化: NBSP除去・全角スペース変換・連続空白圧縮"""
    if s is None:
        return ""
    s = s.replace("\xa0", " ").replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_soup(url: str) -> BeautifulSoup:
    """HTTPリクエスト + エンコーディング検出 + HTML解析"""
    r = requests.get(url, headers=HEADERS, timeout=30)
    ctype = r.headers.get("Content-Type", "").lower()
    if "shift_jis" in ctype or "cp932" in ctype or "ms932" in ctype:
        r.encoding = "cp932"
    elif not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding or "utf-8"
    return BeautifulSoup(r.text, "html.parser")


def find_data_table(soup: BeautifulSoup) -> Tag | None:
    """ページ内のデータテーブルを特定（ヘッダー一致度で判定）"""
    candidate = None
    best_score = -1
    for tbl in soup.find_all("table"):
        text = clean_text(tbl.get_text(" "))
        score = sum(1 for h in EXPECTED_HEADERS if h in text)
        if score > best_score:
            best_score = score
            candidate = tbl
    if candidate and best_score >= len(EXPECTED_HEADERS) // 2:
        return candidate
    return None


def extract_rows_from_table(table: Tag) -> list[list[str]]:
    """テーブルをセルテキストのリストに変換"""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row = [clean_text(c.get_text(" ")) for c in cells]
        if any(cell for cell in row):
            rows.append(row)
    return rows


def align_columns(
    rows: list[list[str]],
) -> tuple[tuple[list[str], list[list[str]]], dict[str, int | None]]:
    """ヘッダー行を検出し、列位置マッピングを返す"""
    header_idx = 0
    best_score = -1
    for i, row in enumerate(rows):
        score = sum(1 for cell in row if cell in EXPECTED_HEADERS)
        if score > best_score:
            best_score = score
            header_idx = i

    header_row = rows[header_idx]
    col_positions: dict[str, int | None] = {}
    for label in EXPECTED_HEADERS:
        try:
            col_positions[label] = header_row.index(label)
        except ValueError:
            col_positions[label] = None

    data_rows = rows[header_idx + 1 :]
    return ([*EXPECTED_HEADERS], data_rows), col_positions


def get_data_rows(url: str) -> list[list[str]]:
    """URLからデータ行を抽出"""
    soup = get_soup(url)
    table = find_data_table(soup)
    if not table:
        raise RuntimeError(
            "Failed to locate data table; the page structure may have changed."
        )
    rows = extract_rows_from_table(table)
    (_, data_rows), _ = align_columns(rows)
    return data_rows


def parse_price_range(s: str) -> tuple[float | None, float | None]:
    """価格範囲文字列 '12 ～ 15' を (low, high) に変換（万円単位）"""
    s = clean_text(s)
    m = re.findall(r"([0-9]+(?:\.[0-9]+)?)", s)
    if len(m) == 1:
        v = float(m[0])
        return v, v
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    return None, None


def parse_cassette(row1: list[str], row2: list[str]) -> dict:
    """2行ペアを解析して車両情報 dict を返す"""
    price_low, price_high = parse_price_range(row1[7])
    data = {
        "maker": row1[0],
        "title": row1[1],
        "grade": row2[1],
        "year": row1[2],
        "mileage_km": float(row1[4]) * 10000,
        "cartype": row1[5],
        "valuation": row1[6],
        "price_low": price_low,
        "price_high": price_high,
        "color": row2[4],
    }
    return data


def extract_car_info(url: str, max_cars: int = 100) -> list[dict]:
    """単一ページから車両情報を抽出"""
    cars: list[dict] = []
    try:
        data_rows = get_data_rows(url)
    except (RuntimeError, requests.RequestException):
        logger.exception("Failed to extract data from %s", url)
        return cars

    for i in range(min(max_cars, len(data_rows) // 2)):
        try:
            row1 = data_rows[2 * i + 7]
            row2 = data_rows[2 * i + 8]
            car = parse_cassette(row1, row2)
            cars.append(car)
        except (IndexError, ValueError):
            logger.warning("Failed to parse row pair %d", i)
            break

    logger.info("Extracted %d cars from %s", len(cars), url)
    return cars


def scrape_aucsupport(
    url: str,
    iterations: int = 5,
    output_path: str | Path | None = None,
) -> list[dict]:
    """複数回スクレイピングして車両リストを返す"""
    all_cars: list[dict] = []
    for i in range(iterations):
        logger.info("Iteration %d: %s", i + 1, url)
        got = extract_car_info(url)
        all_cars.extend(got)
        logger.info("Total cars so far: %d", len(all_cars))
        time.sleep(REQUEST_INTERVAL_SEC)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(all_cars, f)
        logger.info("Saved %d cars to %s", len(all_cars), output_path)

    return all_cars


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scrape_aucsupport(
        url=(
            "https://www.aucsupport.com/soubalist.aspx"
            "?MAKER=%e3%83%88%e3%83%a8%e3%82%bf"
            "&CARNAME=%e3%82%b7%e3%82%a8%e3%83%b3%e3%82%bf"
            "&SYEAR=2010&EYEAR=2025"
        ),
        iterations=5,
        output_path="out/raw/aucsupport_sienta.pkl",
    )
