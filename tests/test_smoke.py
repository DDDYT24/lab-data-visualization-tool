def test_import_app() -> None:
    import app  # noqa: F401


def test_import_main_has_module() -> None:
    import importlib

    main = importlib.import_module("main")
    assert hasattr(main, "__file__")
