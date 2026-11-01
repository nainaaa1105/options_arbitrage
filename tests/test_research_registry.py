import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_registry


class RegistryTests(unittest.TestCase):
    def test_registry_hash_change_fails_closed(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);data=root/'artifact.bin';data.write_bytes(b'original')
            manifest=root/'registry.json'
            manifest.write_text(json.dumps({'stage':'stage','artifacts':{'artifact.bin':research_registry.sha256(data)},'source':{}}))
            with patch.object(research_registry,'ROOT',root):
                self.assertEqual(research_registry.verify(manifest),root/'stage')
                data.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    research_registry.verify(manifest)

    def test_default_legacy_clis_require_explicit_opt_in(self):
        root=Path(__file__).resolve().parents[1]
        for entry in ('train_models.py','generate_signals.py','generate_trade_signals.py'):
            result=subprocess.run([sys.executable,str(root/'src'/entry)],cwd=root,capture_output=True,text=True,timeout=180)
            self.assertNotEqual(result.returncode,0,entry)
            self.assertIn('--allow-legacy-artifacts',result.stderr,entry)


if __name__=='__main__':unittest.main()
