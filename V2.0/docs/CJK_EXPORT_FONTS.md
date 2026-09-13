# LabViz CJK export fonts

LabViz exports figures with Matplotlib on the local API process. A browser font does not
automatically make a PNG, SVG, or PDF contain Chinese glyphs, so the export path resolves a
CJK-capable font before rendering any figure that contains CJK text.

## V2.2 strategy

- The redistributable target is **Noto Sans SC**, distributed under the SIL Open Font License
  1.1. A future Windows installer should ship the required font file and its license notice in
  the installer payload; it must not redistribute Microsoft YaHei, SimSun, PingFang, or another
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

The test renders the same Chinese-labelled chart as PNG, SVG, and PDF and fails if a Matplotlib
missing-glyph warning is emitted. The current Windows verification environment resolved
`Noto Sans SC` from `C:\Windows\Fonts\NotoSansSC-VF.ttf`; that is machine evidence only, not a
claim that the future installer is complete.

Installer work remains responsible for bundling an OFL-compliant font, preserving the license
notice, and repeating this test on a clean machine without relying on Windows fonts.
