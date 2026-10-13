set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
# JVM implicit null checks and safepoint machinery legitimately use these signals.
# Before native exit, the existing ErrorFile diagnostic remains authoritative.
handle SIGSEGV nostop noprint pass
handle SIGBUS nostop noprint pass
# Unqualified "exit" also matches HotSpot's C++ monitor-exit methods during startup.
break -qualified exit
commands
  silent
  printf "Entering native exit: enable fatal-signal backtraces.\n"
  handle SIGSEGV stop print pass
  handle SIGBUS stop print pass
  continue
end
run
thread apply all bt 12
info sharedlibrary
