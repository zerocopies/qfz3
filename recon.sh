#!/bin/bash
# recon.sh — read-only state audit for the Zero Copies workspaces
set +e
echo "=== 1. WORKSPACE LOCATIONS ==="
find ~ -maxdepth 3 -type d \( -name "qfz3*" -o -iname "*quantum*" -o -name "z1-core" -o -name "buzz-router" \) -not -path "*/target/*" 2>/dev/null

echo; echo "=== 2. GIT REMOTES + TRACKED FILES (~/qfz3) ==="
cd ~/qfz3 2>/dev/null && git remote -v && echo "--- buzz-router tracked? ---" && git ls-files | grep -i buzz | head -20

echo; echo "=== 3. BUZZ-ROUTER IN PUBLIC HISTORY? ==="
cd ~/qfz3 2>/dev/null && git log --all --oneline -- buzz-router/ | head -10 && echo "(empty = never committed = safe)"

echo; echo "=== 4. KEY LEAK SCAN (history) ==="
cd ~/qfz3 2>/dev/null && git log -p --all 2>/dev/null | grep -cEi 'gsk_[A-Za-z0-9]{20}|sk-ant-[A-Za-z0-9]|AIza[0-9A-Za-z_-]{30}' && echo "^ match count (0 = clean)"

echo; echo "=== 5. WHICH ENGINE IS CANONICAL? ==="
for f in $(find ~ -path "*/src/graph.rs" -not -path "*/target/*" -not -path "*/vendor/*" 2>/dev/null); do
  echo "--- $f ---"
  grep -cE "KvSnapshot|snapshot_prefix|restore_prefix" "$f" | xargs echo "  v5.1 markers:"
  grep -cE "begin_begin_request" "$f" | xargs echo "  session-boundary markers:"
  stat -c "  modified: %y" "$f"
done

echo; echo "=== 6. IS generate_sync STILL MOCKED? ==="
for f in $(find ~ -path "*/src/engine.rs" -not -path "*/target/*" 2>/dev/null); do
  echo "--- $f ---"; grep -n "Mock" "$f" | head -3
done

echo; echo "=== 7. SWAP ACTIVE? ==="
swapon --show; free -h | head -2

echo; echo "=== 8. PORT 7474 ==="
ss -tlnp 2>/dev/null | grep 7474 || echo "port free"
