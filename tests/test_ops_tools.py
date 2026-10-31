from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_offline_tests as runner
import clean_install_check as clean


class DiscoveryTests(unittest.TestCase):
    def write(self, text):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'test_x.py'
        path.write_text(text)
        return path

    def test_import_time_network_and_credential_calls_are_detected(self):
        bad = self.write('import requests\nresponse = requests.get("https://example.invalid")\n')
        self.assertEqual(runner.import_time_calls(bad), ['get'])
        env = self.write('from dotenv import load_dotenv\nload_dotenv()\n')
        self.assertEqual(runner.import_time_calls(env), ['load_dotenv'])

    def test_guarded_and_pure_modules_are_safe(self):
        good = self.write('"""doc"""\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n'
                          'def main():\n    import requests\n    requests.get("x", timeout=1)\nclass T:\n    x = open\n'
                          'if __name__ == "__main__":\n    main()\n')
        self.assertEqual(runner.import_time_calls(good), [])

    def test_repository_has_no_unguarded_online_modules(self):
        safe, excluded = runner.classify([p for folder in (ROOT / 'tests', ROOT / 'src') for p in folder.glob('test*.py')])
        self.assertEqual([(p.name, c) for p, c in excluded], [])
        self.assertGreater(len(safe), 15)

    def test_every_blocking_http_call_in_tests_has_a_timeout(self):
        import ast
        missing = []
        for path in list((ROOT / 'tests').glob('*.py')) + list((ROOT / 'src').glob('test_*.py')):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Call) and ast.unparse(node.func) in ('requests.get', 'requests.post') \
                        and not any(k.arg == 'timeout' for k in node.keywords):
                    missing.append((path.name, node.lineno))
        self.assertEqual(missing, [])

    def test_network_block_refuses_connections_and_blanks_the_token(self):
        original_connect, original_ex = socket.socket.connect, socket.socket.connect_ex
        import os
        original_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
        try:
            runner.block_network()
            with self.assertRaisesRegex(RuntimeError, 'Network access is disabled'):
                socket.socket().connect(('127.0.0.1', 9))
            self.assertEqual(os.environ['UPSTOX_ACCESS_TOKEN'], '')
        finally:
            socket.socket.connect, socket.socket.connect_ex = original_connect, original_ex
            if original_token is None:
                os.environ.pop('UPSTOX_ACCESS_TOKEN', None)
            else:
                os.environ['UPSTOX_ACCESS_TOKEN'] = original_token


class CleanInstallPlanTests(unittest.TestCase):
    def test_plan_is_ordered_pure_and_never_runs_tests_in_dry_run(self):
        files = [Path('requirements.txt'), Path('requirements/deep-hedging-lock.txt')]
        full = clean.build_commands(Path('/tmp/venv'), files)
        self.assertEqual(full[0][1:3], ['-m', 'venv'])
        self.assertTrue(any(c[-1].endswith('run_offline_tests.py') for c in full))
        self.assertTrue(any('check' in c for c in full))
        dry = clean.build_commands(Path('/tmp/venv'), files, dry_run=True)
        self.assertTrue(all('--dry-run' in c for c in dry if '-r' in c))
        self.assertFalse(any(c[-1].endswith('run_offline_tests.py') for c in dry))

    def test_print_plan_executes_nothing(self):
        self.assertEqual(clean.main(['--print-plan']), 0)


if __name__ == '__main__':
    unittest.main()
