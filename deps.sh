#!/usr/bin/env bash

PKG_LIST=$(fgrep 'using' *.jl -h | sed 's/:.*//g; s/\n/ /g; s/,//g; s/using //g; s/ /\n/g; s/\..*//g' | sort | uniq | tail -n +1)

while read p; do

	echo "using Pkg; Pkg.add(\"$p\")" | julia

done < <(echo "$PKG_LIST")


