# LabViz CJK export fonts

LabViz exports figures with Matplotlib on the local API process. A browser font does not
automatically make a PNG, SVG, or PDF contain Chinese glyphs, so the export path resolves a
CJK-capable font before rendering any figure that contains CJK text.

## V2.2 strategy

- The redistributable target is **Noto Sans SC**, distributed under the SIL Open Font License
  1.1. The repository now includes the font and license in `V2.0/assets/fonts/`; Windows
  candidate staging copies both and package validation requires both. It must not redistribute Microsoft YaHei, SimSun, PingFang, or another
  operating-system font.
- A developer or installer may select an explicit licensed font with the `LABVIZ_CJK_FONT_PATH`
  environment variable. The path should point to a local `.ttf`, `.otf`, or supported `.ttc`
  file whose redistribution terms the packager has verified.
- The local resolver first checks `V2.0/assets/fonts/`, then common Noto font locations. On
  Windows it also checks the installed Noto Sans SC font and OS fonts as a local-development
  fallback. These OS fallbacks are not installer assets and do not establish cross-platform
  packaging support.
- When CJK text is detected, the selected CJK family is applied to the complete figure so Latin
  and CJK labels use one predictable font. English-only figures keep the user's Arial or Times
  New Roman choice.
- If a CJK label is requested and no usable CJK font is available, the API fails with
  `cjk-font-unavailable` and an actionable message. It never silently claims a successful export
  while Matplotlib is dropping glyphs.

## Local verification

From `V2.0/api`, using the repository's Python 3.12/3.13 environment:

```powershell
& .venv\Scripts\python.exe -c "from labviz_api.processing import _cjk_font_info; print(_cjk_font_info())"
& .venv\Scripts\python.exe -m pytest tests/test_v22_samples.py::test_cjk_exports_use_a_cjk_capable_font_without_missing_glyph_warnings -q
```

The test restricts the resolver to the bundled asset, renders the same Chinese-labelled chart
as PNG, SVG, and PDF, and fails on missing-glyph warnings. SVG retains text nodes and embeds
a subset font as a data URL. PDF embeds TrueType glyphs and a Unicode map. Structural tests
check these properties; they do not replace visual inspection on other operating systems.

Clean-machine installer testing remains open. Restricting the resolver in a regression test
demonstrates independence from system fonts, not a completed clean-Windows installation test.
