"""爬虫モジュールのテスト"""

from unittest.mock import MagicMock, patch
import pickle
import tempfile
from pathlib import Path

import pytest
import requests as requests_lib
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
    clean_text as mobilico_clean_text,
    clean_price as mobilico_clean_price,
    extract_money_li,
    norm_text as mobilico_norm_text,
    parse_exhibited_article,
    pick_from_price_block,
    pick_from_spec_block,
    extract_car_info as mobilico_extract_car_info,
    scrape_mobilico,
)
from data_augmentation.websites.aucsupport import (
    EXPECTED_HEADERS,
    align_columns,
    clean_text as aucsupport_clean_text,
    extract_car_info as aucsupport_extract_car_info,
    extract_rows_from_table,
    find_data_table,
    get_soup,
    parse_cassette as aucsupport_parse_cassette,
    parse_price_range,
    scrape_aucsupport,
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


# ---------------------------------------------------------------------------
# AucSupport clean_text Tests (#33)
# ---------------------------------------------------------------------------
class TestAucsupportCleanText:
    """aucsupport.clean_text のテスト"""

    def test_fullwidth_space_and_multiple_spaces(self):
        """#33: 全角空格和连续空白処理"""
        result = aucsupport_clean_text("トヨタ\u3000シエンタ   G")
        assert result == "トヨタ シエンタ G"

    def test_nbsp_removal(self):
        """NBSP を半角スペースに変換"""
        result = aucsupport_clean_text("hello\xa0world")
        assert result == "hello world"

    def test_none_returns_empty(self):
        """None → 空文字列"""
        result = aucsupport_clean_text(None)
        assert result == ""

    def test_strips_whitespace(self):
        """前後空白トリム"""
        result = aucsupport_clean_text("  hello  ")
        assert result == "hello"


# ---------------------------------------------------------------------------
# AucSupport find_data_table Tests (#26)
# ---------------------------------------------------------------------------
class TestAucsupportFindDataTable:
    """find_data_table のテスト"""

    def test_finds_table_with_expected_headers(self):
        """#26: 正确识别包含预期表头的数据表格"""
        headers = " ".join(EXPECTED_HEADERS)
        html = f"""
        <html><body>
            <table id="nav"><tr><td>ナビ</td></tr></table>
            <table id="data">
                <tr><th>{("</th><th>".join(EXPECTED_HEADERS))}</th></tr>
                <tr><td>トヨタ</td><td>シエンタ</td></tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = find_data_table(soup)
        assert result is not None
        assert result.get("id") == "data"

    def test_returns_none_when_no_matching_table(self):
        """表头不匹配时返回 None"""
        html = """
        <html><body>
            <table><tr><td>unrelated</td></tr></table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = find_data_table(soup)
        assert result is None


# ---------------------------------------------------------------------------
# AucSupport extract_rows_from_table Tests (#35)
# ---------------------------------------------------------------------------
class TestAucsupportExtractRows:
    """extract_rows_from_table のテスト"""

    def test_skips_empty_rows(self):
        """#35: 空行をスキップ"""
        html = """
        <table>
            <tr><td>トヨタ</td><td>シエンタ</td></tr>
            <tr><td></td><td></td></tr>
            <tr><td>ホンダ</td><td>フィット</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        rows = extract_rows_from_table(table)
        assert len(rows) == 2
        assert rows[0][0] == "トヨタ"
        assert rows[1][0] == "ホンダ"

    def test_extracts_th_and_td(self):
        """th と td 両方抽出"""
        html = """
        <table>
            <tr><th>メーカー</th><th>車種</th></tr>
            <tr><td>トヨタ</td><td>シエンタ</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        rows = extract_rows_from_table(table)
        assert len(rows) == 2
        assert rows[0] == ["メーカー", "車種"]


# ---------------------------------------------------------------------------
# AucSupport align_columns Tests (#36)
# ---------------------------------------------------------------------------
class TestAucsupportAlignColumns:
    """align_columns のテスト"""

    def test_maps_column_positions(self):
        """#36: 正确映射列位置"""
        rows = [
            ["メーカー", "車種", "年式", "シフト", "走行", "型式", "評価", "価格(万円)", "更新年月", "グレード", "排気量", "車検", "色", "装備"],
            ["トヨタ", "シエンタ", "2020", "CVT", "3.5", "NHP170G", "4.5", "120 ～ 150", "2024/01", "G", "1500", "2025/11", "白", "ナビ"],
        ]
        (_, data_rows), col_positions = align_columns(rows)
        assert col_positions["メーカー"] == 0
        assert col_positions["車種"] == 1
        assert col_positions["価格(万円)"] == 7
        assert len(data_rows) == 1

    def test_missing_columns_mapped_to_none(self):
        """#36: 缺失列映射为 None"""
        rows = [
            ["メーカー", "車種", "年式"],
            ["トヨタ", "シエンタ", "2020"],
        ]
        (_, data_rows), col_positions = align_columns(rows)
        assert col_positions["メーカー"] == 0
        assert col_positions["シフト"] is None
        assert col_positions["走行"] is None


# ---------------------------------------------------------------------------
# AucSupport parse_cassette Tests (#27, #28, #29, #38, #39)
# ---------------------------------------------------------------------------
def _build_aucsupport_row_pair(
    maker: str = "トヨタ",
    title: str = "シエンタ",
    grade: str = "G",
    year: str = "2020",
    shift: str = "CVT",
    mileage: str = "5.2",
    cartype: str = "NHP170G",
    valuation: str = "4.5",
    price: str = "120 ～ 150",
    update_date: str = "2024/01",
    displacement: str = "1500",
    inspection: str = "2025/11",
    color: str = "白",
    equipment: str = "ナビ",
) -> tuple[list[str], list[str]]:
    """构建 AucSupport 的 2 行数据对"""
    row1 = [maker, title, year, shift, mileage, cartype, valuation, price, update_date]
    row2 = ["", grade, "", "", color, "", "", "", ""]
    return row1, row2


class TestAucsupportParseCassette:
    """parse_cassette のテスト"""

    def test_extract_maker(self):
        """#27: 正确提取制造商字段"""
        row1, row2 = _build_aucsupport_row_pair(maker="トヨタ")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["maker"] == "トヨタ"

    def test_extract_title(self):
        """#28: 正确提取车种名称"""
        row1, row2 = _build_aucsupport_row_pair(title="シエンタ")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["title"] == "シエンタ"

    def test_extract_grade(self):
        """#29: 正确提取等级信息（来自 row2）"""
        row1, row2 = _build_aucsupport_row_pair(grade="FUNBASE G")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["grade"] == "FUNBASE G"

    def test_mileage_multiplied_by_10000(self):
        """#38: 走行距离正确乘以 10000"""
        row1, row2 = _build_aucsupport_row_pair(mileage="5.2")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["mileage_km"] == 52000.0

    def test_extract_color(self):
        """#39: 正确提取颜色字段（来自 row2[4]）"""
        row1, row2 = _build_aucsupport_row_pair(color="パールホワイト")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["color"] == "パールホワイト"

    def test_extract_price_range(self):
        """价格范围正确解析"""
        row1, row2 = _build_aucsupport_row_pair(price="120 ～ 150")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["price_low"] == 120.0
        assert result["price_high"] == 150.0

    def test_extract_year_and_cartype(self):
        """年式と型式の抽出"""
        row1, row2 = _build_aucsupport_row_pair(year="2020", cartype="NHP170G")
        result = aucsupport_parse_cassette(row1, row2)
        assert result["year"] == "2020"
        assert result["cartype"] == "NHP170G"


# ---------------------------------------------------------------------------
# AucSupport get_soup Tests (#34)
# ---------------------------------------------------------------------------
class TestAucsupportGetSoup:
    """get_soup のテスト"""

    def test_detects_shift_jis_encoding(self):
        """#34: Shift_JIS 编码检测"""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html; charset=shift_jis"}
        mock_resp.text = "<html><body>テスト</body></html>"
        mock_resp.encoding = None
        mock_resp.apparent_encoding = "utf-8"

        with patch("data_augmentation.websites.aucsupport.requests.get", return_value=mock_resp):
            soup = get_soup("https://example.com")

        assert mock_resp.encoding == "cp932"
        assert soup is not None

    def test_detects_cp932_encoding(self):
        """Content-Type に cp932 を含む場合"""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html; charset=cp932"}
        mock_resp.text = "<html><body>テスト</body></html>"
        mock_resp.encoding = None

        with patch("data_augmentation.websites.aucsupport.requests.get", return_value=mock_resp):
            soup = get_soup("https://example.com")

        assert mock_resp.encoding == "cp932"

    def test_falls_back_to_apparent_encoding(self):
        """encoding が iso-8859-1 の場合 apparent_encoding にフォールバック"""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = "<html><body>テスト</body></html>"
        mock_resp.encoding = "iso-8859-1"
        mock_resp.apparent_encoding = "utf-8"

        with patch("data_augmentation.websites.aucsupport.requests.get", return_value=mock_resp):
            soup = get_soup("https://example.com")

        assert mock_resp.encoding == "utf-8"


# ---------------------------------------------------------------------------
# AucSupport extract_car_info Tests (#37)
# ---------------------------------------------------------------------------
class TestAucsupportExtractCarInfo:
    """extract_car_info のテスト"""

    def test_runtime_error_returns_empty_list(self):
        """#37: 表格未找到时返回空列表"""
        with patch(
            "data_augmentation.websites.aucsupport.get_data_rows",
            side_effect=RuntimeError("no table"),
        ):
            result = aucsupport_extract_car_info("https://example.com")
        assert result == []

    def test_request_exception_returns_empty_list(self):
        """网络异常返回空列表"""
        with patch(
            "data_augmentation.websites.aucsupport.get_data_rows",
            side_effect=requests_lib.RequestException("timeout"),
        ):
            result = aucsupport_extract_car_info("https://example.com")
        assert result == []


# ---------------------------------------------------------------------------
# AucSupport scrape_aucsupport Tests
# ---------------------------------------------------------------------------
class TestAucsupportPagination:
    """scrape_aucsupport テスト"""

    def test_iterations_and_pickle_save(self):
        """複数回イテレーションと pickle 保存"""
        car_data = [{"maker": "トヨタ", "title": "シエンタ"}]
        with patch(
            "data_augmentation.websites.aucsupport.extract_car_info",
            return_value=car_data,
        ):
            with patch("data_augmentation.websites.aucsupport.time.sleep"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = Path(tmpdir) / "aucsupport_test.pkl"
                    result = scrape_aucsupport(
                        url="https://example.com",
                        iterations=3,
                        output_path=out_path,
                    )
                    assert len(result) == 3  # 3 iterations × 1 car
                    assert out_path.exists()
                    with open(out_path, "rb") as f:
                        loaded = pickle.load(f)
                    assert loaded == result


# ---------------------------------------------------------------------------
# Mobilico Helper — mock HTML builder
# ---------------------------------------------------------------------------
def _build_mobilico_article_html(
    title: str = "トヨタ シエンタ FUNBASE G",
    total_price: str = "68.0",
    base_price: str = "55.0",
    other_fee: str = "13.0",
    specs: dict | None = None,
) -> BeautifulSoup:
    """構建 Mobilico 単一出品記事の mock HTML (article タグ)"""
    if specs is None:
        specs = {
            "年式": "2019年",
            "走行距離": "3.5万km",
            "車検": "26年8月",
            "板金歴": "なし",
            "出品地域": "東京都",
            "予定納期": "2週間以内",
        }

    price_items = ""
    for label, val in [
        ("支払総額", total_price),
        ("本体価格", base_price),
        ("諸費用", other_fee),
    ]:
        price_items += f"""
        <li class="meta-item">
            <span class="label">{label}</span>
            <strong>{val}</strong>
            <small>万円</small>
        </li>
        """

    spec_items = ""
    for label, val in specs.items():
        spec_items += f"""
        <li class="meta-item">
            <span class="label">{label}</span>
            <span class="text-sm">{val}</span>
        </li>
        """

    html = f"""
    <article class="exhibited-car card">
        <header><h3 class="card-title">{title}</h3></header>
        <div class="card-body">
            <div class="meta-items">
                <ul>
                    {price_items}
                </ul>
            </div>
            <div class="meta-items">
                <ul>
                    {spec_items}
                </ul>
            </div>
        </div>
    </article>
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("article")


def _build_mobilico_price_ul(
    total: str = "68.0",
    base: str = "55.0",
    fee: str = "13.0",
) -> BeautifulSoup:
    """構建 Mobilico 価格 UL ブロック"""
    html = f"""
    <ul>
        <li class="meta-item">
            <span class="label">支払総額</span>
            <strong>{total}</strong><small>万円</small>
        </li>
        <li class="meta-item">
            <span class="label">本体価格</span>
            <strong>{base}</strong><small>万円</small>
        </li>
        <li class="meta-item">
            <span class="label">諸費用</span>
            <strong>{fee}</strong><small>万円</small>
        </li>
    </ul>
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("ul")


def _build_mobilico_spec_ul(specs: dict | None = None) -> BeautifulSoup:
    """構建 Mobilico スペック UL ブロック"""
    if specs is None:
        specs = {
            "年式": "2019年",
            "走行距離": "3.5万km",
            "車検": "26年8月",
            "板金歴": "なし",
            "出品地域": "東京都",
            "予定納期": "2週間以内",
        }
    items = ""
    for label, val in specs.items():
        items += f"""
        <li class="meta-item">
            <span class="label">{label}</span>
            <span class="text-sm">{val}</span>
        </li>
        """
    html = f"<ul>{items}</ul>"
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("ul")


# ---------------------------------------------------------------------------
# Mobilico parse_exhibited_article Tests (#15)
# ---------------------------------------------------------------------------
class TestMobilicoParseExhibitedArticle:
    """parse_exhibited_article 各字段提取テスト"""

    def test_extract_title(self):
        """#15: 正确提取车辆标题"""
        article = _build_mobilico_article_html(title="トヨタ シエンタ FUNBASE G")
        result = parse_exhibited_article(article)
        assert result["title"] == "トヨタ シエンタ FUNBASE G"
        assert result["title"] is not None
        assert len(result["title"]) > 0

    def test_extract_title_empty_header(self):
        """header が空の場合 title は None"""
        html = """
        <article class="exhibited-car card">
            <header></header>
            <div class="card-body"></div>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        result = parse_exhibited_article(article)
        assert result["title"] is None

    def test_non_article_raises_value_error(self):
        """#24: 非 article 标签抛出 ValueError"""
        html = "<div class='test'>test</div>"
        soup = BeautifulSoup(html, "html.parser")
        div_tag = soup.find("div")
        with pytest.raises(ValueError, match="article"):
            parse_exhibited_article(div_tag)

    def test_full_field_extraction(self):
        """完整字段提取验证"""
        article = _build_mobilico_article_html()
        result = parse_exhibited_article(article)
        assert result["title"] == "トヨタ シエンタ FUNBASE G"
        assert result["total_price"] == 68.0
        assert result["base_price"] == 55.0
        assert result["other_fee"] == 13.0
        assert result["year"] == "2019年"
        assert result["mileage_km"] == 35000.0
        assert result["inspection"] == "202608"
        assert result["bodywork_history"] == "なし"
        assert result["region"] == "東京都"


# ---------------------------------------------------------------------------
# Mobilico pick_from_price_block Tests (#16, #17, #18)
# ---------------------------------------------------------------------------
class TestMobilicoPickFromPriceBlock:
    """pick_from_price_block のテスト"""

    def test_extract_total_price(self):
        """#16: 正确提取支払総額"""
        ul = _build_mobilico_price_ul(total="68.0")
        result = pick_from_price_block(ul)
        assert result["total_price"] == 68.0
        assert result["total_price"] > 0

    def test_extract_base_price(self):
        """#17: 正确提取本体价格"""
        ul = _build_mobilico_price_ul(base="55.0")
        result = pick_from_price_block(ul)
        assert result["base_price"] == 55.0
        assert result["base_price"] > 0

    def test_extract_other_fee(self):
        """#18: 正确提取诸费用"""
        ul = _build_mobilico_price_ul(fee="13.0")
        result = pick_from_price_block(ul)
        assert result["other_fee"] == 13.0
        assert result["other_fee"] > 0

    def test_empty_block(self):
        """空の価格ブロック"""
        html = "<ul></ul>"
        soup = BeautifulSoup(html, "html.parser")
        ul = soup.find("ul")
        result = pick_from_price_block(ul)
        assert result["total_price"] is None
        assert result["base_price"] is None
        assert result["other_fee"] is None


# ---------------------------------------------------------------------------
# Mobilico pick_from_spec_block Tests (#19, #20, #21, #22)
# ---------------------------------------------------------------------------
class TestMobilicoPickFromSpecBlock:
    """pick_from_spec_block のテスト"""

    def test_extract_year(self):
        """#19: 正确提取年式"""
        ul = _build_mobilico_spec_ul(specs={"年式": "2019年"})
        result = pick_from_spec_block(ul)
        assert result["year"] == "2019年"

    def test_extract_mileage_with_man_km(self):
        """#20: 正确提取走行距离并转换「万km」单位"""
        ul = _build_mobilico_spec_ul(specs={"走行距離": "3.5万km"})
        result = pick_from_spec_block(ul)
        assert result["mileage_km"] == 35000.0

    def test_extract_mileage_without_man(self):
        """走行距離が万km単位でない場合"""
        ul = _build_mobilico_spec_ul(specs={"走行距離": "8000km"})
        result = pick_from_spec_block(ul)
        assert result["mileage_km"] == 8000.0

    def test_extract_inspection_date(self):
        """#21: 正确解析车检日期「26年8月」→ '202608'"""
        ul = _build_mobilico_spec_ul(specs={"車検": "26年8月"})
        result = pick_from_spec_block(ul)
        assert result["inspection"] == "202608"

    def test_extract_inspection_single_digit_month(self):
        """車検: 1桁月のゼロパディング"""
        ul = _build_mobilico_spec_ul(specs={"車検": "27年3月"})
        result = pick_from_spec_block(ul)
        assert result["inspection"] == "202703"

    def test_extract_inspection_no_match(self):
        """車検: パースできない値は None"""
        ul = _build_mobilico_spec_ul(specs={"車検": "なし"})
        result = pick_from_spec_block(ul)
        assert result["inspection"] is None

    def test_extract_region(self):
        """#22: 正确提取出品地域"""
        ul = _build_mobilico_spec_ul(specs={"出品地域": "大阪府"})
        result = pick_from_spec_block(ul)
        assert result["region"] == "大阪府"

    def test_extract_bodywork_history(self):
        """板金歴の抽出"""
        ul = _build_mobilico_spec_ul(specs={"板金歴": "あり"})
        result = pick_from_spec_block(ul)
        assert result["bodywork_history"] == "あり"

    def test_extract_estimated_delivery(self):
        """予定納期の抽出"""
        ul = _build_mobilico_spec_ul(specs={"予定納期": "2週間以内"})
        result = pick_from_spec_block(ul)
        assert result["estimated_delivery"] == "2週間以内"


# ---------------------------------------------------------------------------
# Mobilico extract_money_li Tests (#23)
# ---------------------------------------------------------------------------
class TestMobilicoExtractMoneyLi:
    """extract_money_li のテスト"""

    def test_combines_strong_and_small(self):
        """#23: <strong> と <small> を結合して '68.0 万円' 形式"""
        html = '<li class="meta-item"><strong>68.0</strong><small>万円</small></li>'
        soup = BeautifulSoup(html, "html.parser")
        li = soup.find("li")
        result = extract_money_li(li)
        assert result == "68.0 万円"

    def test_no_strong_falls_back_to_norm_text(self):
        """<strong> がない場合は norm_text にフォールバック"""
        html = '<li class="meta-item">応談</li>'
        soup = BeautifulSoup(html, "html.parser")
        li = soup.find("li")
        result = extract_money_li(li)
        assert result == "応談"

    def test_strong_only_no_small(self):
        """<small> がない場合でも <strong> の数字を返す"""
        html = '<li class="meta-item"><strong>100</strong></li>'
        soup = BeautifulSoup(html, "html.parser")
        li = soup.find("li")
        result = extract_money_li(li)
        assert "100" in result


# ---------------------------------------------------------------------------
# Mobilico extract_car_info Tests
# ---------------------------------------------------------------------------
class TestMobilicoExtractCarInfo:
    """extract_car_info のテスト"""

    def test_http_error_returns_none(self):
        """HTTP 错误返回 None"""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("data_augmentation.websites.mobilico.requests.get", return_value=mock_resp):
            result = mobilico_extract_car_info("https://mobilico.jp/test")
        assert result is None

    def test_request_exception_returns_none(self):
        """网络异常返回 None"""
        with patch(
            "data_augmentation.websites.mobilico.requests.get",
            side_effect=requests_lib.RequestException("timeout"),
        ):
            result = mobilico_extract_car_info("https://mobilico.jp/test")
        assert result is None

    def test_success_returns_list(self):
        """正常返回车辆列表"""
        article = _build_mobilico_article_html(title="テスト車両")
        full_html = f"<html><body>{article}</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = full_html.encode("utf-8")
        with patch("data_augmentation.websites.mobilico.requests.get", return_value=mock_resp):
            result = mobilico_extract_car_info("https://mobilico.jp/test")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["title"] == "テスト車両"


# ---------------------------------------------------------------------------
# Mobilico scrape_mobilico Tests (#25)
# ---------------------------------------------------------------------------
class TestMobilicoPagination:
    """scrape_mobilico 分页テスト"""

    def test_page_url_construction(self):
        """#25: 分页抓取正确拼接 ?page=N 参数"""
        urls_called = []

        def mock_extract(url):
            urls_called.append(url)
            return [{"title": f"car_{len(urls_called)}"}]

        with patch("data_augmentation.websites.mobilico.extract_car_info", side_effect=mock_extract):
            with patch("data_augmentation.websites.mobilico.time.sleep"):
                result = scrape_mobilico(
                    base_url="https://mobilico.jp/exhibited_cars/TOYOTA/SIENTA",
                    page_count=4,
                )

        assert "?page=1" in urls_called[0]
        assert "?page=2" in urls_called[1]
        assert "?page=3" in urls_called[2]
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

        with patch("data_augmentation.websites.mobilico.extract_car_info", side_effect=mock_extract):
            with patch("data_augmentation.websites.mobilico.time.sleep"):
                result = scrape_mobilico(
                    base_url="https://mobilico.jp/test",
                    page_count=5,
                )

        assert call_count == 2
        assert len(result) == 1

    def test_saves_pickle(self):
        """结果可保存为 pickle"""
        with patch(
            "data_augmentation.websites.mobilico.extract_car_info",
            return_value=[{"title": "test_car", "year": "2020年"}],
        ):
            with patch("data_augmentation.websites.mobilico.time.sleep"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = Path(tmpdir) / "sub" / "mobilico_test.pkl"
                    result = scrape_mobilico(
                        base_url="https://mobilico.jp/test",
                        page_count=2,
                        output_path=out_path,
                    )
                    assert out_path.exists()
                    with open(out_path, "rb") as f:
                        loaded = pickle.load(f)
                    assert loaded == result
