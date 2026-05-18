param(
  [ValidateSet('smoke', 'paper')]
  [string]$Mode = 'smoke',
  [int]$EvalSamples = 200000
)

$ErrorActionPreference = 'Stop'

Write-Host "Running in mode: $Mode"
Write-Host "Working directory: $(Get-Location)"

if ($Mode -eq 'smoke') {
  Write-Host "[1/3] Quick eval (reduced samples)"
  python .\eval.py --samples $EvalSamples

  Write-Host "[2/3] Quick 11-round key-recovery sanity check"
  python .\test_key_recovery.py --quick

  Write-Host "[3/3] Quick key-rank stats sample"
  python .\key_rank.py --start-exp 5 --end-exp 6
} else {
  Write-Host "[1/3] Full eval from repository defaults"
  python .\eval.py --samples 1000000

  Write-Host "[2/3] Full key-recovery suite (11r + 12r)"
  python .\test_key_recovery.py --runs-11r 100 --runs-12r 20

  Write-Host "[3/3] Key-rank statistics sweep"
  python .\key_rank.py --start-exp 5 --end-exp 8
}

Write-Host "Finished."
