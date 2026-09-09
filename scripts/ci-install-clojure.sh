#!/usr/bin/env bash
# Pinned CLI payload from Clojure's official distribution, without the upstream
# installer's unchecked GitHub-release download. Run as root for /usr/local, or
# pass an owned absolute prefix for an isolated smoke test.
set -euo pipefail

cli_version=1.12.0.1488
cli_sha256=bc19be0010bef0421c26fd3bec7bc3bca08c192828d59a151845422dc4420742
cli_prefix=${1:-/usr/local}
if [[ ! "$cli_prefix" =~ ^/[a-zA-Z0-9/._-]+$ ]]; then
  echo "unsupported Clojure installation prefix: $cli_prefix" >&2
  exit 2
fi

cli_work=$(mktemp -d -t raster-clojure-install.XXXXXX)
trap 'rm -rf -- "$cli_work"' EXIT
cli_archive="$cli_work/clojure-tools.tar.gz"
curl --fail --show-error --location --retry 3 --retry-delay 2 \
  --connect-timeout 20 --max-time 180 \
  "https://download.clojure.org/install/clojure-tools-${cli_version}.tar.gz" \
  --output "$cli_archive"
printf '%s  %s\n' "$cli_sha256" "$cli_archive" | sha256sum --check --status
gzip -t "$cli_archive"
tar xzf "$cli_archive" -C "$cli_work"

cli_source="$cli_work/clojure-tools"
cli_lib="$cli_prefix/lib/clojure"
cli_bin="$cli_prefix/bin"
cli_man="$cli_prefix/share/man/man1"
install -d "$cli_lib/libexec" "$cli_bin" "$cli_man"
install -m644 "$cli_source/deps.edn" "$cli_source/example-deps.edn" "$cli_source/tools.edn" "$cli_lib/"
install -m644 "$cli_source/exec.jar" "$cli_source/clojure-tools-${cli_version}.jar" "$cli_lib/libexec/"
sed -i "s@PREFIX@$cli_lib@g" "$cli_source/clojure"
sed -i "s@BINDIR@$cli_bin@g" "$cli_source/clj"
install -m755 "$cli_source/clojure" "$cli_source/clj" "$cli_bin/"
install -m644 "$cli_source/clojure.1" "$cli_source/clj.1" "$cli_man/"
"$cli_bin/clojure" -Sdescribe
