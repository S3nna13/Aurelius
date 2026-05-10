#!/bin/bash
# Merge all dependabot branches into current branch
set -e

BRANCHES=$(git branch -r | grep dependabot | sed 's|origin/||')
CONFLICTS=0
MERGED=0
SKIPPED=0

echo "Found $(echo "$BRANCHES" | wc -l | tr -d ' ') dependabot branches"
echo "---"

for branch in $BRANCHES; do
    echo -n "Merging $branch... "
    
    # Check if branch already merged
    if git merge-base --is-ancestor origin/$branch HEAD 2>/dev/null; then
        echo "ALREADY MERGED (skipping)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Attempt merge
    if git merge --no-edit origin/$branch 2>/dev/null; then
        echo "OK"
        MERGED=$((MERGED + 1))
    else
        echo "CONFLICT - resolving (accepting incoming)"
        # Get conflicted files
        CONFLICTED_FILES=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
        if [ -z "$CONFLICTED_FILES" ]; then
            # No conflicts in tracked files, just commit
            git commit --no-edit 2>/dev/null || true
            echo "  (auto-resolved)"
            MERGED=$((MERGED + 1))
        else
            # Accept incoming (theirs) for all conflicts
            for f in $CONFLICTED_FILES; do
                if [ -f "$f" ]; then
                    git checkout --theirs "$f" 2>/dev/null || true
                fi
            done
            git add -A
            git commit --no-edit -m "merge: $branch (conflicts resolved - accepting dependabot changes)" 2>/dev/null || true
            CONFLICTS=$((CONFLICTS + 1))
        fi
    fi
done

echo "---"
echo "Summary: $MERGED merged, $CONFLICTS with conflicts, $SKIPPED already merged"
