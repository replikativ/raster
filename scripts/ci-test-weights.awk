# Input: source bytes TAB test path. Optional timings: path TAB positive integer ms.
# Unknown/new files retain a size estimate calibrated to the measured files in this checkout.
BEGIN {
  if (timings != "") {
    while ((status = (getline line < timings)) > 0) {
      if (line ~ /^#/ || line == "") continue
      n = split(line, fields, "\t")
      if (n != 2 || fields[1] == "" || fields[2] !~ /^[1-9][0-9]*$/ ||
          fields[1] in costs) {
        print "invalid or duplicate CI timing row: " line > "/dev/stderr"
        bad = 1
        exit 2
      }
      costs[fields[1]] = fields[2] + 0
    }
    if (status < 0) {
      print "cannot read CI timings: " timings > "/dev/stderr"
      bad = 1
      exit 2
    }
    close(timings)
  }
}
{
  paths[NR] = $2
  bytes[NR] = $1
  if ($2 in costs) {
    known_bytes += $1
    known_ms += costs[$2]
  }
}
END {
  if (bad) exit 2
  for (i = 1; i <= NR; i++) {
    estimate = paths[i] in costs ? costs[paths[i]] :
      (known_bytes > 0 ? int(bytes[i] * known_ms / known_bytes + 0.5) : bytes[i])
    if (estimate < 1) estimate = 1
    printf "%.0f\t%s\n", estimate, paths[i]
  }
}
