#!/bin/bash
# What exit code does a wq host see when a train_ladder.sh-shaped script gets SIGTERM?
# child.py imitates Lightning: catches SIGTERM, finishes the step (2 s), raises SystemExit() (SIGTERMException) -> 0.
D=$(mktemp -d)
cat > $D/child.py <<'PY'
import signal, sys, time
class SIGTERMException(SystemExit): pass
got = []
signal.signal(signal.SIGTERM, lambda s, f: got.append(s))
for i in range(100):
    time.sleep(0.1)
    if got:
        time.sleep(2); raise SIGTERMException
sys.exit(7)
PY
cat > $D/job.sh <<EOS
#!/bin/bash
set -o pipefail
trap 'echo "[job.sh] EXIT trap"' EXIT     # env.sh installs an EXIT trap like this
python $D/child.py
rc=\$?
echo "main.py exit \$rc"
exit \$rc
EOS
echo "== A: timeout runs the job, an external TERM is sent to timeout (fetch_pending on_signal / wq cancel)"
( exec timeout --kill-after=30 600s bash $D/job.sh ) & p=$!; sleep 1; kill -TERM $p; wait $p; echo "host sees rc=$?"
echo "== B: timeout expires (host end)"
( exec timeout --kill-after=30 1s bash $D/job.sh ) & p=$!; wait $p; echo "host sees rc=$?"
echo "== C: TERM to the whole process group of a plain bash job (Slurm scancel/time limit)"
setsid bash -c "bash $D/job.sh; echo \"slurm sees rc=\$?\"" & sleep 1; pkill -TERM -f "$D/job.sh"; pkill -TERM -f "$D/child.py"; wait
rm -rf $D
