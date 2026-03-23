"""爬虫モジュールのテスト"""

from unittest.mock import MagicMock, patch
import pickle
import tempfile
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from data_augmentation.websites.carsensor import (
    clean_price,
    clean_text,
    extract_car_info,
    norm_text,
    parse_cassette,
    scrape_carsensor,
)
from data_augmentation.websites.mobilico import (
    extract_money_li,
    pick_from_price_block,
)
from data_augmentation.websites.aucsupport import (
    parse_price_range,
    find_data_table,
)


def _build_cassette_html(
    title: str = "トヨタ シエンタ G",
    specs: dict | None = None,
    total_price: str = "189",
    base_price: str = "170",
    region_parts: list[str] | None = None,
) -> BeautifulSoup:
    """构建 Carsensor 单个车辆卡片的 mock HTML"""
    if specs is None:
        specs = {
            "年式": "2020(R2)年",
            "走行距離": "5.2万km",
            "車検": "2026年11月",
            "修復歴": "なし",
            "保証": "あり",
            "整備": "整備付",
            "排気量": "1500cc",
            "ミッション": "CVT",
        }
    if region_parts is None:
        region_parts = ["東京都", "世田谷区"]

    spec_items = ""
    for label, value in specs.items():
        spec_items += f"""
        <div class="specList__detailBox">
            <dt class="specList__title">{label}</dt>
            <dd class="specList__data">{value}</dd>
        </div>
        """

    region_ps = "".join(f"<p>{p}</p>" for p in region_parts)

    html = f"""
    <div class="cassette js_listTableCassette">
        <div class="cassetteMain__title"><a>{title}</a></div>
        <div class="cassetteMain__specInfo">
            <dl class="specList">
                {spec_items}
            </dl>
        </div>
        <div class="totalPrice__content">{total_price}</div>
        <div class="basePrice__content">{base_price}</div>
        <div class="cassetteSub__area">
            {region_ps}
        </div>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div", class_="cassette")


# ---------------------------------------------------------------------------
# Carsensor Utils Tests
# ---------------------------------------------------------------------------
class TestCarsensorUtils:
    def test_clean_text_removes_nbsp(self):
        assert clean_text("hello\xa0world") == "hello world"

    def test_clean_text_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_clean_text_none_returns_none(self):
        assert clean_text(None) is None

    def test_clean_text_empty_returns_none(self):
        assert clean_text("") is None

    def test_clean_price_extracts_integer(self):
        assert clean_price("189 万円") == 189.0

    def test_clean_price_extracts_decimal(self):
        assert clean_price("68.5 万円") == 68.5

    def test_clean_price_non_string_returns_none(self):
        assert clean_price(123) is None

    def test_clean_price_no_number_returns_none(self):
        assert clean_price("応談") is None


# ---------------------------------------------------------------------------
# Carsensor parse_cassette Tests (feature_list #0-#9, #121)
# ---------------------------------------------------------------------------
class TestCarsensorParseCassette:
    """parse_cassette 对各字段的提取测试"""

    def test_extract_title(self):
        """#0: 正确提取车辆标题"""
        card = _build_cassette_html(title="トヨタ シエンタ G")
        result = parse_cassette(card)
        assert result["title"] == "トヨタ シエンタ G"
        assert result["title"] is not None
        assert len(result["title"]) > 0

    def test_extract_year(self):
        """#1: 正确提取年式字段（4 位整数）"""
        card = _build_cassette_html(specs={"年式": "2020(R2)年"})
        result = parse_cassette(card)
        assert result["year"] == 2020
        assert isinstance(result["year"], int)
        assert 1900 <= result["year"] <= 2100

    def test_extract_mileage_with_unit_conversion(self):
        """#2: 正确提取走行距离并转换「万km」单位"""
        card = _build_cassette_html(specs={"走行距離": "5.2万km"})
        result = parse_cassette(card)
        assert result["mileage_km"] == 52000.0

    def test_extract_mileage_without_man(self):
        """走行距離が万km単位でない場合"""
        card = _build_cassette_html(specs={"走行距離": "8000km"})
        result = parse_cassette(card)
        assert result["mileage_km"] == 8000.0

    def test_extract_inspection_date(self):
        """#3: 正确提取车检日期"""
        card = _build_cassette_html(specs={"車検": "2026年11月"})
        result = parse_cassette(card)
        assert result["inspection"] == "202611"

    def test_extract_inspection_single_digit_month(self):
        """车检日期：单位数月份补零"""
        card = _build_cassette_html(specs={"車検": "2025年3月"})
        result = parse_cassette(card)
        assert result["inspection"] == "202503"

    def test_extract_accident_history(self):
        """#4: 正确提取修复历史"""
        card = _build_cassette_html(specs={"修復歴": "なし"})
        result = parse_cassette(card)
        assert result["accident_history"] == "なし"

    def test_extract_accident_history_yes(self):
        """修复历史：有"""
        card = _build_cassette_html(specs={"修復歴": "あり"})
        result = parse_cassette(card)
        assert result["accident_history"] == "あり"

    def test_extract_total_price(self):
        """#5: 正确提取支払総額"""
        card = _build_cassette_html(total_price="189")
        result = parse_cassette(card)
        assert result["total_price"] == 189.0
        assert result["total_price"] > 0

    def test_extract_base_price(self):
        """#6: 正确提取車両本体価格"""
        card = _build_cassette_html(base_price="170.5")
        result = parse_cassette(card)
        assert result["base_price"] == 170.5
        assert result["base_price"] > 0

    def test_extract_region(self):
        """#7: 正确提取地域信息"""
        card = _build_cassette_html(region_parts=["東京都", "世田谷区"])
        result = parse_cassette(card)
        assert "東京都" in result["region"]

    def test_extract_displacement(self):
        """#8: 正确提取排气量"""
        card = _build_cassette_html(specs={"排気量": "1500cc"})
        result = parse_cassette(card)
        assert result["displacement"] is not None
        assert "1500" in result["displacement"]

    def test_extract_transmission(self):
        """#9: 正确提取变速箱类型"""
        card = _build_cassette_html(specs={"ミッション": "CVT"})
        result = parse_cassette(card)
        assert result["transmission"] is not None
        assert result["transmission"] == "CVT"

    def test_full_field_extraction(self):
        """#121: parse_cassette 完整字段提取验证"""
        card = _build_cassette_html()
        result = parse_cassette(card)

        assert result["title"] == "トヨタ シエンタ G"
        assert result["year"] == 2020
        assert result["mileage_km"] == 52000.0
        assert result["inspection"] == "202611"
        assert result["accident_history"] == "なし"
        assert result["warranty"] == "あり"
        assert result["maintenance"] == "整備付"
        assert result["displacement"] == "1500cc"
        assert result["transmission"] == "CVT"
        assert result["total_price"] == 189.0
        assert result["base_price"] == 170.0
        assert "東京都" in result["region"]


# ---------------------------------------------------------------------------
# Carsensor extract_car_info Tests (#13)
# ---------------------------------------------------------------------------
class TestCarsensorExtractCarInfo:
    """extract_car_info のテスト"""

    def test_http_error_returns_none(self):
        """#13: HTTP 错误返回 None"""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("data_augmentation.websites.carsensor.requests.get", return_value=mock_resp):
            result = extract_car_info("https://example.com/test.html")
        assert result is None

    def test_request_exception_returns_none(self):
        """网络异常返回 None"""
        import requests

        with patch(
            "data_augmentation.websites.carsensor.requests.get",
            side_effect=requests.RequestException("timeout"),
        ):
            result = extract_car_info("https://example.com/test.html")
        assert result is None

    def test_success_returns_list(self):
        """正常返回车辆列表"""
        card = _build_cassette_html(title="テスト車両")
        full_html = f"<html><body>{card}</body></html>"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = full_html.encode("utf-8")

        with patch("data_augmentation.websites.carsensor.requests.get", return_value=mock_resp):
            result = extract_car_info("https://example.com/test.html")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["title"] == "テスト車両"


# ---------------------------------------------------------------------------
# Carsensor scrape_carsensor Tests (#14)
# ---------------------------------------------------------------------------
class TestCarsensorPagination:
    """scrape_carsensor 分页测试"""

    def test_page_url_construction(self):
        """#14: 分页抓取正确拼接 URL"""
        urls_called = []

        def mock_extract(url):
            urls_called.append(url)
            return [{"title": f"car_{len(urls_called)}"}]

        with patch("data_augmentation.websites.carsensor.extract_car_info", side_effect=mock_extract):
            with patch("data_augmentation.websites.carsensor.time.sleep"):
                result = scrape_carsensor(
                    base_url="https://www.carsensor.net/usedcar/index.html",
                    page_count=4,
                )

        # 第 1 页使用原始 URL
        assert urls_called[0] == "https://www.carsensor.net/usedcar/index.html"
        # 第 2 页替换 .html 为 2.html
        assert urls_called[1] == "https://www.carsensor.net/usedcar/index2.html"
        # 第 3 页替换 .html 为 3.html
        assert urls_called[2] == "https://www.carsensor.net/usedcar/index3.html"
        # 共调用 3 次（page_count=4 → range(1,4)）
        assert len(urls_called) == 3
        assert len(result) == 3

    def test_stops_on_none(self):
        """提取失败时停止爬取"""
        call_count = 0

        def mock_extract(url):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return None
            return [{"title": "car"}]

        with patch("data_augmentation.websites.carsensor.extract_car_info", side_effect=mock_extract):
            with patch("data_augmentation.websites.carsensor.time.sleep"):
                result = scrape_carsensor(
                    base_url="https://example.com/index.html",
                    page_count=5,
                )

        assert call_count == 2
        assert len(result) == 1

    def test_saves_pickle(self):
        """结果可保存为 pickle"""
        with patch(
            "data_augmentation.websites.carsensor.extract_car_info",
            return_value=[{"title": "test_car", "year": 2020}],
        ):
            with patch("data_augmentation.websites.carsensor.time.sleep"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = Path(tmpdir) / "sub" / "test.pkl"
                    result = scrape_carsensor(
                        base_url="https://example.com/index.html",
                        page_count=2,
                        output_path=out_path,
                    )
                    assert out_path.exists()
                    with open(out_path, "rb") as f:
                        loaded = pickle.load(f)
                    assert loaded == result


# ---------------------------------------------------------------------------
# AucSupport Utils Tests
# ---------------------------------------------------------------------------
class TestAucsupportUtils:
    def test_parse_price_range_two_values(self):
        low, high = parse_price_range("12 ～ 15")
        assert low == 12.0
        assert high == 15.0

    def test_parse_price_range_single_value(self):
        low, high = parse_price_range("10")
        assert low == 10.0
        assert high == 10.0

    def test_parse_price_range_empty(self):
        low, high = parse_price_range("")
        assert low is None
        assert high is None
