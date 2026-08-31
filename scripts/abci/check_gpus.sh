N=4
R=ngpus

pbsnodes -a | awk -v R="$R" -v N="$N" '
function flush() {
  if (node != "") {
    free_gpu = avail - used
    if (avail > 0 && free_gpu >= N && state !~ /(offline|down|unknown|stale)/) {
      printf "%-30s free_gpu=%d total_gpu=%d used_gpu=%d state=%s\n", node, free_gpu, avail, used, state
      count++
    }
  }
}

/^[^[:space:]]+$/ {
  flush()
  node=$1
  state=""
  avail=0
  used=0
  next
}

$1 == "state" && $2 == "=" {
  state=$3
}

$1 == "resources_available." R && $2 == "=" {
  avail=$3
}

$1 == "resources_assigned." R && $2 == "=" {
  used=$3
}

END {
  flush()
  print "----"
  print "nodes_with_free_gpu_ge_" N "=" count
}'