# 快速测试脚本：验证 train-all 全流程
# 使用极小参数快速跑通逻辑

$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting full coverage smoke test..."

# Construct the command as a single string to avoid backtick issues
$cmd = "python main.py train-all " +
"--db data/ssq.db " +
"--recent 150 " +
"--cat-window 10 " +
"--cat-iter 5 " +
"--cat-depth 4 " +
"--cat-fresh " +
"--seq-window 10 " +
"--seq-epochs 1 " +
"--seq-fresh " +
"--run-tft " +
"--tft-window 20 " +
"--tft-epochs 1 " +
"--tft-batch 32 " +
"--tft-fresh " +
"--run-nhits " +
"--nhits-input 30 " +
"--nhits-steps 10 " +
"--nhits-fresh " +
"--run-timesnet " +
"--timesnet-input 30 " +
"--timesnet-steps 10 " +
"--timesnet-fresh " +
"--run-prophet " +
"--prophet-fresh " +
"--run-blend " +
"--blend-train 60 " +
"--blend-test 20 " +
"--blend-step 20"

Write-Host "Running command: $cmd"
Invoke-Expression $cmd
