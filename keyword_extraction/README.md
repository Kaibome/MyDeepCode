# Multilingual Keyword Extraction

This module extracts 3-5 business keywords from a single delivery review.

## Supported Languages

- Indonesian (`id`)
- Thai (`th`)
- Vietnamese (`vi`)
- Portuguese (`pt`)

## API

```python
from keyword_extraction import extract_keywords

keywords = extract_keywords(
    "Makanannya enak, pengiriman cepat, harga murah.",
    top_k=5,
    enable_online_fallback=False,
)
```

## Output Keywords

The normalized output is constrained to:

- `味道`
- `配送速度`
- `价格`
- `包装`
- `分量`
- `服务`

## Behavior

- Runs offline by default: cleaning -> language detection -> YAKE -> business mapping.
- Optional online fallback (`enable_online_fallback=True`) tries translation and re-extraction when local results are too short.
- If online translation fails, it returns offline results without raising errors.
