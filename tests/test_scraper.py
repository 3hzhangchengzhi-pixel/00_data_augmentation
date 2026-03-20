"""爬虫モジュールのテスト"""

import pytest

from data_augmentation.websites.carsensor import (
    clean_price,
    clean_text,
    norm_text,
)
from data_augmentation.websites.mobilico import (
    extract_money_li,
    pick_from_price_block,
)
from data_augmentation.websites.aucsupport import (
    parse_price_range,
    find_data_table,
)


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
