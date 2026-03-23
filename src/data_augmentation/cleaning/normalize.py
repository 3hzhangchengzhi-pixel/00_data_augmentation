"""字段標準化モジュール

価格・里程・年式・車検日期・修復歴・変速箱・地域の標準化処理。
"""

import re


# 和暦→西暦の変換テーブル
_ERA_MAP: dict[str, int] = {
    "令和": 2018,
    "平成": 1988,
    "昭和": 1925,
    "大正": 1911,
    "明治": 1867,
    "R": 2018,
    "H": 1988,
    "S": 1925,
    "T": 1911,
    "M": 1867,
}


def normalize_price(price_man: float | None) -> int | None:
    """万円単位の価格を日元整数に変換"""
    if price_man is None:
        return None
    return int(price_man * 10000)


def normalize_mileage(mileage_km: float | None) -> int | None:
    """里程をキロメートル整数に統一"""
    if mileage_km is None:
        return None
    return int(mileage_km)


def normalize_year(year_str: str | int | None) -> int | None:
    """年式を西暦整数に統一（和暦対応）

    対応形式:
      - 整数: 2020
      - 西暦文字列: "2020年", "2020(R2)年", "2020"
      - 和暦文字列: "令和2年", "R2年", "平成30年", "H30年"
    """
    if year_str is None:
        return None
    if isinstance(year_str, int):
        return year_str

    s = str(year_str).strip()

    # 西暦4桁が先頭にある場合（"2020(R2)年" など）
    m = re.match(r"(\d{4})", s)
    if m:
        return int(m.group(1))

    # 和暦パターン: "令和2年", "R2年" など
    for era, base in _ERA_MAP.items():
        pattern = rf"{re.escape(era)}\s*(\d+)"
        m = re.search(pattern, s)
        if m:
            return base + int(m.group(1))

    return None


def normalize_inspection(value: str | None) -> str | None:
    """車検日期を 'YYYY-MM' 形式に標準化

    対応形式: '202611', '2026-11', '2026年11月'
    """
    if value is None:
        return None
    s = str(value).strip()

    # YYYYMM 形式
    m = re.match(r"^(\d{4})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # YYYY-MM 形式（そのまま）
    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    # YYYY年MM月 形式
    m = re.match(r"(\d{4})年(\d{1,2})月", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    return None


def normalize_accident_history(value: str | None) -> bool | None:
    """修復歴をブール値に変換"""
    if value is None:
        return None
    value = value.strip()
    if value in ("あり", "有", "有り"):
        return True
    if value in ("なし", "無", "無し"):
        return False
    return None


def normalize_transmission(value: str | None) -> str | None:
    """変速箱タイプを AT/MT/CVT に統一"""
    if value is None:
        return None
    v = value.strip()
    v_upper = v.upper()
    if "CVT" in v_upper:
        return "CVT"
    if "AT" in v_upper or "オートマ" in v:
        return "AT"
    if "MT" in v_upper or "マニュアル" in v:
        return "MT"
    return v


def normalize_region(value: str | None) -> str | None:
    """地域名を都道府県のみに統一

    '東京都 渋谷区' → '東京都'
    '東京 渋谷' → '東京都'
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # スペース区切りの場合、最初の部分を取得
    parts = re.split(r"[\s　]+", s)
    prefecture = parts[0]

    # 都道府県サフィックスの補完
    if not re.search(r"(都|道|府|県)$", prefecture):
        # 北海道は特別
        if prefecture == "北海道":
            pass
        elif prefecture in ("東京",):
            prefecture += "都"
        elif prefecture in ("大阪", "京都"):
            prefecture += "府"
        else:
            prefecture += "県"

    return prefecture
