def test_import_app_has_main():
    import app  # noqa: F401
    assert hasattr(app, "main")
