"""Carsensor (carsensor.net) 爬虫模块

日本最大の中古車情報サイトから車両データを取得する。
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


def clean_text(s: str | None) -> str | None:
    """テキストの正規化: NBSP除去・前後空白トリム"""
    if not s:
        return None
    return s.replace("\xa0", " ").strip()


def norm_text(el: Tag | None) -> str | None:
    """BS4 要素内の全テキストを空白区切りで結合"""
    if not el:
        return None
    return " ".join(el.stripped_strings)


def clean_price(s: str | int | float | None) -> float | None:
    """価格文字列から数値を抽出（万円単位）"""
    if not isinstance(s, str):
        return None
    s_clean = s.replace(" ", "")
    match = re.search(r"\d+(\.\d+)?", s_clean)
    if match:
        return float(match.group())
    return None


def parse_cassette(card: Tag) -> dict:
    """単一の車両カセット（掲載カード）を解析して dict を返す"""
    data: dict = {}

    for box in card.select(
        ".cassetteMain__specInfo dl.specList .specList__detailBox"
    ):
        dt_el = box.select_one(".specList__title")
        dd_el = box.select_one(".specList__data")
        if not dt_el or not dd_el:
            continue
        label = dt_el.get_text(strip=True)
        value = norm_text(dd_el)

        if label == "年式" and value:
            m = re.match(r"^\d{4}", value)
            if m:
                data["year"] = int(m.group())
        elif label == "走行距離" and value:
            m = re.match(r"^\d+(\.\d+)?", value)
            if m:
                mileage = float(m.group())
                if "万km" in value:
                    mileage *= 10000.0
                data["mileage_km"] = mileage
        elif label == "車検" and value:
            m = re.match(r"(\d{4}).*?(\d{1,2})月", value)
            if m:
                data["inspection"] = m.group(1) + m.group(2).zfill(2)
            else:
                data["inspection"] = None
        elif label == "修復歴":
            data["accident_history"] = value
        elif label == "保証":
            data["warranty"] = value
        elif label == "整備":
            data["maintenance"] = value
        elif label == "排気量":
            data["displacement"] = value
        elif label == "ミッション":
            data["transmission"] = value

    # 支払総額(万円)
    total = card.select_one(".totalPrice__content")
    data["total_price"] = clean_price(norm_text(total))

    # 車両本体価格(万円)
    base = card.select_one(".basePrice__content")
    data["base_price"] = clean_price(norm_text(base))

    # 車名タイトル
    title = card.select_one(".cassetteMain__title a")
    data["title"] = clean_text(norm_text(title))

    # 販売地域（都道府県 + 市区町村）
    area_ps = card.select(".cassetteSub__area p")
    data["region"] = " ".join(norm_text(p) or "" for p in area_ps).strip()

    return data


def extract_car_info(url: str) -> list[dict] | None:
    """単一ページから全車両情報を抽出"""
    try:
        res = requests.get(url, timeout=30)
        if res.status_code != 200:
            logger.warning("HTTP %d for %s", res.status_code, url)
            return None
    except requests.RequestException:
        logger.exception("Request failed for %s", url)
        return None

    soup = BeautifulSoup(res.content, "html.parser")
    car_divs = soup.find_all("div", class_=["cassette", "js_listTableCassette"])
    cars = [parse_cassette(div) for div in car_divs]
    logger.info("Extracted %d cars from %s", len(cars), url)
    return cars


def scrape_carsensor(
    base_url: str,
    page_count: int = 11,
    output_path: str | Path | None = None,
) -> list[dict]:
    """複数ページをスクレイピングして車両リストを返す"""
    all_cars: list[dict] = []
    for page in range(1, page_count):
        if page > 1:
            url = base_url.replace(".html", f"{page}.html")
        else:
            url = base_url
        logger.info("Fetching page %d: %s", page, url)
        got = extract_car_info(url)
        if got is None:
            logger.warning("Stopping at page %d (no data)", page)
            break
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
    scrape_carsensor(
        base_url="https://www.carsensor.net/usedcar/bTO/s077/index.html",
        page_count=11,
        output_path="out/raw/carsensor_sienta.pkl",
    )
