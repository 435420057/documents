# s

Do this in cases when you dont want to change the os-level settings, but only want to disable the OOM killer for a single process. This is useful when youre on a shared machine/server.

The OOM killer uses the process level metric called `oom_score_adj` to decide if/when to kill a process. 
This file is present in `/proc/$pid/oom_score_adj`. The oom_score_adj can vary from `-1000` to `1000`, by default it is 0.

You can add a large negative score to this file to reduce the probability of your process getting picked and terminated by OOM killer. 
When you set it to -1000, it can use 100% memory and still avoid getting terminated by OOM killer.

On the other hand, if you assign 1000 to it, the Linux kernel will keep killing the process even if it uses minimal memory.


Run this command
`sudo echo -1000 > /proc/<PID>/oom_score_adj`

We need root/sudo rights as Linux does not allow normal users to **reduce** the OOM score. You can **increase** the OOM score as a normal user without any special permissions.

Reference:https://dev.to/rrampage/surviving-the-linux-oom-killer-2ki9


# Dealing with it at the OS level

This disables the oom killer at the os level and avoids doing the previously mentioned steps for each process. Do this if you are on a machine/server that you are using specifically for your application, for example on a cloud vm for a given job.

Details from: https://serverfault.com/a/142003/316820

By default Linux has a somewhat brain-damaged concept of memory management: it lets you allocate more memory than your system has, then randomly shoots a process in the head when it gets in trouble. (The actual semantics of what gets killed are more complex than that - Google "Linux OOM Killer" for lots of details and arguments about whether it's a good or bad thing).

To disable this behaviour:

1. Disable the OOM Killer (Put `vm.oom-kill = 0` in `/etc/sysctl.conf`)
2. Disable memory overcommit (Put `vm.overcommit_memory = 2` in `/etc/sysctl.conf`)
   Note that this is a trinary value: 0 = "estimate if we have enough RAM", 1 = "Always say yes", 2 = "say no if we don't have the memory")

These settings will make Linux behave in the traditional way (if a process requests more memory than is available malloc() will fail and the process requesting the memory is expected to cope with that failure).

Reboot your machine to make it reload `/etc/sysctl.conf`, or use the proc file system to enable right away, without reboot:

`echo 2 > /proc/sys/vm/overcommit_memory` s