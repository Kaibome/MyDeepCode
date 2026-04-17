from keyword_extraction import extract_keywords
from keyword_extraction import extractor


def _assert_keyword_constraints(result: list[str]) -> None:
    assert isinstance(result, list)
    assert 1 <= len(result) <= 5
    assert any(
        keyword in {"味道", "配送速度", "价格", "包装", "分量", "服务"} for keyword in result
    )


def test_extract_keywords_indonesian_positive() -> None:
    text = "Makanannya enak, pengiriman cepat, harga murah dan kemasan rapi."
    result = extract_keywords(text)
    _assert_keyword_constraints(result)
    assert "味道" in result


def test_extract_keywords_thai_negative() -> None:
    text = "อาหารไม่อร่อย ส่งช้า ราคาแพง แต่พนักงานสุภาพ"
    result = extract_keywords(text)
    _assert_keyword_constraints(result)
    assert "配送速度" in result or "价格" in result


def test_extract_keywords_vietnamese_mixed() -> None:
    text = "Đồ ăn ngon nhưng giao hàng chậm, giá hơi đắt, đóng gói ổn."
    result = extract_keywords(text)
    _assert_keyword_constraints(result)
    assert "味道" in result


def test_extract_keywords_portuguese_mixed() -> None:
    text = "Comida saborosa, entrega rápida, mas o preço está alto."
    result = extract_keywords(text)
    _assert_keyword_constraints(result)
    assert "配送速度" in result or "价格" in result


def test_short_text_does_not_fail() -> None:
    result = extract_keywords("bom")
    assert isinstance(result, list)
    assert len(result) <= 5


def test_online_fallback_works_when_enabled(monkeypatch) -> None:
    def fake_translate(_: str, source_lang: str) -> str:
        assert source_lang in {"en", "id", "th", "vi", "pt"}
        return "delicious food and fast delivery"

    monkeypatch.setattr(extractor, "_translate_for_fallback", fake_translate)
    monkeypatch.setattr(extractor, "_extract_offline", lambda *_args, **_kwargs: [])
    result = extract_keywords("teks pendek", enable_online_fallback=True)
    assert isinstance(result, list)


def test_online_fallback_failure_returns_offline_result(monkeypatch) -> None:
    monkeypatch.setattr(extractor, "_translate_for_fallback", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(extractor, "_extract_offline", lambda *_args, **_kwargs: ["价格"])
    result = extract_keywords("teks pendek", enable_online_fallback=True)
    assert result == ["价格"]
