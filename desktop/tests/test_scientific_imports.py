"""Deferred desktop imports remain thread-safe and use the original library functions."""
import importlib.util
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

source=Path(__file__).resolve().parents[2]/'V2.0/api/labviz_api/scientific_imports.py'
spec=importlib.util.spec_from_file_location('desktop_scientific_imports',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ScientificImportTests(unittest.TestCase):
    def test_parallel_first_use_imports_once_and_returns_original_results(self):
        import math
        lazy=module.DeferredModule('math')
        with patch.object(module.importlib,'import_module',return_value=math) as loader:
            with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(lambda _:lazy.sqrt(81),range(40)))
            self.assertEqual(values,[9.0]*40);loader.assert_called_once_with('math')
            self.assertIs(lazy.sqrt,math.sqrt)

    def test_browser_mode_keeps_eager_import_behavior(self):
        import math
        with patch.dict(os.environ,{'LABVIZ_DESKTOP_MODE':'0'}):self.assertIs(module.scientific_module('math'),math)
        with patch.dict(os.environ,{'LABVIZ_DESKTOP_MODE':'1'}):self.assertIsInstance(module.scientific_module('math'),module.DeferredModule)

    def test_failed_import_is_reported_and_can_be_retried(self):
        import math
        lazy=module.DeferredModule('math')
        with patch.object(module.importlib,'import_module',side_effect=[ImportError('test failure'),math]):
            with self.assertRaises(ImportError):lazy.sqrt(4)
            self.assertEqual(lazy.sqrt(4),2)

    def test_library_attribute_patching_preserves_the_real_module_semantics(self):
        import math
        lazy=module.DeferredModule('math')
        lazy.desktop_test_value=123
        self.assertEqual(math.desktop_test_value,123)
        del lazy.desktop_test_value
        self.assertFalse(hasattr(math,'desktop_test_value'))
