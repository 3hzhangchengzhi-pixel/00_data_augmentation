"""Mobilico (mobilico.jp) 爬虫模块

二手車取引プラットフォームから車両データを取得する。
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


def extract_money_li(li: Tag) -> str | None:
    """
    価格系 <li> から '68.0 万円' のような文字列を組み立てる。
    <strong> 内の数字と <small> 内の単位を結合。
    """
    strong = li.select_one("strong")
    if not strong:
        return norm_text(li)
    number = "".join(strong.stripped_strings)
    unit = norm_text(li.select_one("small")) or ""
    number = clean_text(number) or ""
    unit = clean_text(unit) or ""
    return f"{number} {unit}".strip()


def pick_from_price_block(price_block: Tag) -> dict:
    """価格ULから支払総額/本体価格/諸費用を抽出"""
    out: dict = {"total_price": None, "base_price": None, "other_fee": None}
    for li in price_block.select("li.meta-item"):
        label = norm_text(li.select_one(".label"))
        if not label:
            continue
        val = extract_money_li(li)
        if label == "支払総額":
            out["total_price"] = clean_price(val)
        elif label == "本体価格":
            out["base_price"] = clean_price(val)
        elif label == "諸費用":
            out["other_fee"] = clean_price(val)
    return out


def pick_from_spec_block(spec_block: Tag) -> dict:
    """スペックULから車両情報を抽出"""
    out: dict = {
        "year": None,
        "mileage_km": None,
        "inspection": None,
        "bodywork_history": None,
        "region": None,
        "estimated_delivery": None,
    }

    for li in spec_block.select("li.meta-item"):
        label = norm_text(li.select_one(".label"))
        value = norm_text(li.select_one(".text-sm"))

        if label == "年式":
            out["year"] = value
        elif label == "走行距離" and value:
            m = re.match(r"^\d+(\.\d+)?", value)
            if m:
                mileage = float(m.group())
                if "万km" in value:
                    mileage *= 10000.0
                out["mileage_km"] = mileage
        elif label == "車検" and value:
            m = re.match(r"(\d+)年(\d+)月", value)
            if m:
                year_full = 2000 + int(m.group(1))
                month = int(m.group(2))
                out["inspection"] = f"{year_full}{month:02d}"
            else:
                out["inspection"] = None
        elif label == "板金歴":
            out["bodywork_history"] = value
        elif label == "出品地域":
            out["region"] = value
        elif label == "予定納期":
            out["estimated_delivery"] = value

    return out


def parse_exhibited_article(article_tag: Tag) -> dict:
    """単一の出品記事を解析して dict を返す"""
    if article_tag.name != "article":
        raise ValueError("article タグではありません")

    data: dict = {
        "title": None,
        "total_price": None,
        "base_price": None,
        "other_fee": None,
        "year": None,
        "mileage_km": None,
        "inspection": None,
        "bodywork_history": None,
        "region": None,
        "estimated_delivery": None,
    }

    # タイトル
    title_el = article_tag.select_one("header .card-title")
    data["title"] = norm_text(title_el)

    # 価格ブロック
    meta_blocks = article_tag.select("div.card-body > div.meta-items")
    if meta_blocks:
        price_block = meta_blocks[0].select_one("ul")
        if price_block:
            data.update(pick_from_price_block(price_block))

    # スペックブロック
    if len(meta_blocks) > 1:
        spec_ul = meta_blocks[1].select_one("ul")
        if spec_ul:
            data.update(pick_from_spec_block(spec_ul))

    return data


def extract_car_info(
    url: str = "https://mobilico.jp/exhibited_cars/TOYOTA/SIENTA",
) -> list[dict] | None:
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
    car_divs = soup.find_all(
        "article", class_=["exhibited-car", "exhibited-car-overview", "card"]
    )
    cars = [parse_exhibited_article(div) for div in car_divs]
    logger.info("Extracted %d cars from %s", len(cars), url)
    return cars


def scrape_mobilico(
    base_url: str = "https://mobilico.jp/exhibited_cars/TOYOTA/SIENTA",
    page_count: int = 11,
    output_path: str | Path | None = None,
) -> list[dict]:
    """複数ページをスクレイピングして車両リストを返す"""
    all_cars: list[dict] = []
    for page in range(1, page_count):
        url = f"{base_url}?page={page}"
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
    scrape_mobilico(
        page_count=11,
        output_path="out/raw/mobilico_sienta.pkl",
    )
