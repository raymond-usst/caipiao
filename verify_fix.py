
import sys
import io
import pytorch_lightning as pl

# Capture stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

print("Importing main...")
try:
    import main
except ImportError:
    # Need to add current dir to path
    sys.path.append(".")
    import main
except Exception as e:
    # main might fail to import due to missing args or whatever, but logging setup should run
    print(f"Import main warning: {e}")

print("Calling pl.seed_everything(42)")
pl.seed_everything(42)

# Restore
out = sys.stdout.getvalue()
err = sys.stderr.getvalue()
sys.stdout = old_stdout
sys.stderr = old_stderr

print(f"Stdout len: {len(out)}")
print(f"Stderr len: {len(err)}")
print(f"Stdout: '{out}'")
print(f"Stderr: '{err}'")

if "Seed set to" in out or "Seed set to" in err:
    print("FAILURE: 'Seed set to' message found.")
    sys.exit(1)
else:
    print("SUCCESS: 'Seed set to' message NOT found.")
