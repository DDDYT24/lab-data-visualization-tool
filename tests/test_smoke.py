import importlib


def test_import_main_has_module() -> None:
    main = importlib.import_module("main")
    assert hasattr(main, "__file__")
