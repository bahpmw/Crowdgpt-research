set -x
P=../../.venv/bin/python
$P harness.py adam lr=0.003 --seconds 120 --label base-adam
$P harness.py dust pop=8 chunk=8 lr=0.05 --seconds 120 --label base-dust8
$P harness.py dust pop=32 chunk=16 lr=0.05 --seconds 120 --label base-dust32
$P harness.py mezo k=1 eps=0.001 lr=0.0001 --seconds 120 --label base-mezo
echo done
