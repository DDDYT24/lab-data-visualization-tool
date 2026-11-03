from datetime import datetime, timezone
import unittest
from retention_checks import assert_retained

NOW=datetime(2026,10,25,tzinfo=timezone.utc)
LOCAL={'storageMode':'local','expiresAt':None}
EXPIRED={'storageMode':'temporary-cloud','expiresAt':'2026-10-24T23:00:00Z'}


class RetentionChecksTests(unittest.TestCase):
    def test_only_an_expired_temporary_project_may_disappear(self):
        self.assertEqual(assert_retained({'local':LOCAL,'demo':EXPIRED},{'local':LOCAL},NOW),['demo'])

    def test_local_project_loss_is_rejected_even_with_an_expired_demo(self):
        with self.assertRaisesRegex(AssertionError,'durable or unexpired'):
            assert_retained({'local':LOCAL,'demo':EXPIRED},{},NOW)

    def test_unexpired_or_invalid_expiry_cannot_hide_loss(self):
        for expiry in ['2026-10-25T01:00:00Z','not-a-date',None,'2026-10-24T23:00:00']:
            with self.subTest(expiry=expiry),self.assertRaisesRegex(AssertionError,'durable or unexpired'):
                assert_retained({'demo':{**EXPIRED,'expiresAt':expiry}},{},NOW)

    def test_unexpected_new_project_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Unexpected'):
            assert_retained({'local':LOCAL},{'local':LOCAL,'other':LOCAL},NOW)
