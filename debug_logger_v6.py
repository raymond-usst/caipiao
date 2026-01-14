
import logging
import pytorch_lightning as pl
import sys
import io

# Mock stdout
old_stdout = sys.stdout
sys.stdout = io.StringIO()

print("Calling pl.seed_everything(42)")
pl.seed_everything(42)

output = sys.stdout.getvalue()
sys.stdout = old_stdout

print(f"Captured output length: {len(output)}")
print(f"Captured output: '{output}'")

if "Seed set to" in output:
    print("CONFIRMED: Output is via stdout/print (or logging to stdout)")
else:
    print("OUTPUT NOT CAPTURED via stdout redirection. Might be stderr?")
    
# Check stderr
old_stderr = sys.stderr
sys.stderr = io.StringIO()
pl.seed_everything(42)
err_output = sys.stderr.getvalue()
sys.stderr = old_stderr
print(f"Captured stderr: '{err_output}'")
