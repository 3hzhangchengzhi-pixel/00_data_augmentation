"""字段標準化モジュール

価格・里程・年式・車検日期・修復歴・変速箱・地域の標準化処理。
"""


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
    """年式を西暦整数に統一（和暦対応）"""
    if year_str is None:
        return None
    if isinstance(year_str, int):
        return year_str
    # TODO: 和暦→西暦変換の実装
    try:
        return int(str(year_str)[:4])
    except (ValueError, IndexError):
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
    value = value.upper().strip()
    if "CVT" in value:
        return "CVT"
    if "AT" in value or "オートマ" in value.upper():
        return "AT"
    if "MT" in value or "マニュアル" in value:
        return "MT"
    return value
