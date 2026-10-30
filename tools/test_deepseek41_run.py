"""Offline receipt validation only: no processes, HTTP, GPU, or model reads."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from deepseek41_run import ALIAS, exact_done_metrics, ready_for_exact_run


DONE = 'DONE 64 256 38192.0 7592.7 length 0 0 0 0 0'
FIXTURE = Path(__file__).resolve().parent.parent / 'frontend' / 'tests' / 'data' / 'deepseek41-receipt.json'


class ExactRunTests(unittest.TestCase):
    def test_preserved_run(self):
        # Use the sanitized tracked receipt; never call main(), which starts inference.
        with patch('deepseek41_run.subprocess.Popen', side_effect=AssertionError('process forbidden')), \
             patch('deepseek41_run.request', side_effect=AssertionError('HTTP forbidden')):
            receipt = json.loads(FIXTURE.read_text(encoding='utf-8'))
            metrics = exact_done_metrics(receipt['protocol'])
            saved = receipt['result']
            for key, value in metrics.items():
                self.assertEqual(value, saved[key], key)
            self.assertTrue(ready_for_exact_run(
                receipt['health'], receipt['models']))

    def test_requires_loaded_health_and_alias(self):
        models = {'data': [{'id': ALIAS}]}
        self.assertTrue(ready_for_exact_run({'status': 'ok', 'loaded': True}, models))
        for health in ({'status': 'ok', 'loaded': False}, {'status': 'ok'},
                       {'status': 'loading', 'loaded': True}):
            with self.subTest(health=health):
                self.assertFalse(ready_for_exact_run(health, models))
        for wrong in ({}, {'data': []}, {'data': [{'id': 'other-model'}]}):
            with self.subTest(models=wrong):
                self.assertFalse(ready_for_exact_run({'status': 'ok', 'loaded': True}, wrong))

    def test_rejects_missing_or_ambiguous_done(self):
        for protocol in ('', 'READY 1024 stop\n', DONE + '\n' + DONE):
            with self.subTest(protocol=protocol), self.assertRaises(RuntimeError):
                exact_done_metrics(protocol)

    def test_rejects_incomplete_or_cancelled_run(self):
        for n, pp, finish in ((63, 256, 'length'), (64, 255, 'length'),
                              (3, 256, 'cancel'), (64, 256, 'cancel'), (64, 256, 'stop')):
            with self.subTest(n=n, pp=pp, finish=finish), self.assertRaises(RuntimeError):
                exact_done_metrics(f'DONE {n} {pp} 38192.0 7592.7 {finish} 0 0 0 0 0')

    def test_rejects_nonpositive_or_nonfinite_times(self):
        for bad in ('0', '-1', 'nan', 'inf', '-inf'):
            for field in (3, 4):
                parts = DONE.split()
                parts[field] = bad
                with self.subTest(bad=bad, field=field), self.assertRaises(RuntimeError):
                    exact_done_metrics(' '.join(parts))

    def test_rejects_reused_or_speculative_run(self):
        for field in (6, 7, 8, 9, 10):
            parts = DONE.split()
            parts[field] = '1'
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                exact_done_metrics(' '.join(parts))

    def test_rejects_malformed_done(self):
        for protocol in ('DONE 64', 'DONE x 256 38192.0 7592.7 length 0 0 0 0 0',
                         DONE.rsplit(' ', 1)[0], DONE + ' extra'):
            with self.subTest(protocol=protocol), self.assertRaises(RuntimeError):
                exact_done_metrics(protocol)


if __name__ == '__main__':
    unittest.main()
