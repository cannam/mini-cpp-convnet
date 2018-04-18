#!/bin/bash

set -e

mydir=$(dirname "$0")
cd "$mydir"

program_under_test=flower

if [ -n "$1" ]; then
    program_under_test="$1"
fi    

set -u

total=0
good=0
bad=0

test_dir=data/test

echo "Evaluating $program_under_test..."

for infile in $test_dir/*/*.png ; do
    actual=${infile##$test_dir/}
    actual=${actual%%/*}
    estimated=$(./"$program_under_test" "$infile" 2>/dev/null | fgrep '%' | head -1 | sed 's/:.*//')
    if [ "$actual" = "$estimated" ]; then
        echo "$infile: correct: $actual"
        good=$((good + 1))
    else
        echo "$infile: wrong: $estimated (should be $actual)"
        bad=$((bad + 1))
    fi
    total=$((total + 1))
done

percent=$((good*100/total))
echo
echo "Got $good of $total correct ($percent%)"
