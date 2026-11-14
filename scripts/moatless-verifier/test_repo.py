import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("repo.py")
spec = importlib.util.spec_from_file_location("moatless_repo", MODULE)
repo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repo)


class ApplyPatchSnapshotTest(unittest.TestCase):
    def test_initial_snapshot_preserves_binary_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            blob = root / "state.bin"
            blob.write_bytes(b"before\x00")
            subprocess.run(["git", "add", "state.bin"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            blob.write_bytes(b"after\x00")

            self.assertFalse(repo.apply_patch(None, str(root), {"instance_id": "binary"}))
            snapshot = root.parent / "temp_binary_test_init.patch"
            self.assertTrue(snapshot.exists())
            self.assertIn("GIT binary patch", snapshot.read_text())


if __name__ == "__main__":
    unittest.main()
