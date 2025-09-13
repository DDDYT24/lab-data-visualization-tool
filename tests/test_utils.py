# tests/test_utils.py
import os
import sys

# 让 tests 能导入项目根目录的 app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402
from app import (     # noqa: E402
    validate_columns_exist,
    load_csv_bytes,
)

def test_validate_columns_exist_ok(monkeypatch):
    # 避免单测时真正往 Streamlit 界面输出
    monkeypatch.setattr("app.st.error", lambda *a, **k: None)
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert validate_columns_exist(df, ["a", "b"]) is True

def test_load_csv_bytes_ok():
    # 正确：传入 CSV 的字节串，而不是 None
    csv_bytes = b"a,b\n1,2\n3,4\n"
    df = load_csv_bytes(csv_bytes)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
