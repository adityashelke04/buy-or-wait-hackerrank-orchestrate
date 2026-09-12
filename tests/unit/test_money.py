from decimal import Decimal

from buyorwait.money import dec, fmt_currency, fmt_plain


def test_dec_parses_from_string_without_float_error():
    assert dec("0.1") + dec("0.2") == Decimal("0.3")


def test_dec_returns_none_for_blank():
    assert dec("") is None
    assert dec("   ") is None
    assert dec(None) is None


def test_dec_returns_none_for_garbage():
    assert dec("not a number") is None


def test_fmt_plain_strips_trailing_zeros():
    assert fmt_plain(Decimal("25256.00")) == "25256"
    assert fmt_plain(Decimal("620.40")) == "620.4"
    assert fmt_plain(Decimal("17229139.20")) == "17229139.2"
    assert fmt_plain(Decimal("603.30")) == "603.3"


def test_fmt_plain_keeps_two_decimals_when_they_are_significant():
    assert fmt_plain(Decimal("87170.56")) == "87170.56"
    assert fmt_plain(Decimal("1543.35")) == "1543.35"


def test_fmt_plain_never_uses_scientific_notation():
    assert fmt_plain(Decimal("46018000")) == "46018000"
    assert fmt_plain(Decimal("1000")) == "1000"
    assert fmt_plain(Decimal("0")) == "0"


def test_fmt_currency_uses_thousands_separators():
    assert fmt_currency("ZAR", Decimal("25256")) == "ZAR 25,256"
    assert fmt_currency("IDR", Decimal("29158400")) == "IDR 29,158,400"
    assert fmt_currency("EUR", Decimal("1543.35")) == "EUR 1,543.35"


def test_fmt_currency_keeps_two_decimals_for_sub_unit_values():
    assert fmt_currency("EUR", Decimal("620.40")) == "EUR 620.40"
    assert fmt_currency("USD", Decimal("23.5")) == "USD 23.50"
