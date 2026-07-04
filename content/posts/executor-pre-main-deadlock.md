+++
title = "The time our executor started up and then did nothing"
description = "Debugging a pre-main startup hang: jemalloc's profiler init, libgcc's unwinder, and Antithesis's coverage library conspire into a self-deadlock"
date = 2026-07-02

[taxonomies]
tags = ["debugging", "jemalloc", "kubernetes", "antithesis"]

[extra]
toc = true
comment = false
+++

I was hesitant to write this down, mainly because I wasn't sure if the approaches I took were in accordance to standard/established procedures in debugging this sort of thing. So, I want to position this more in the form of an anecdote and less of something instructional or "this is a cool debugging situation that you can learn from". However, I think the problem was general enough that other people might find it useful. And more importantly, something that I can refer to in the future.

This is a walkthrough of how I found and fixed a startup hang in our query engine at e6data. I have tried to keep the commands, their output, and what they mean together so that the reasoning is easy to follow.

(Needless to say, I will be obfuscating a few details because company IP, but will attempt to convey the intent without needing the details :P)

## Some Background

This was during the time we were still setting up our query engine to work in [Antithesis](https://antithesis.com), a testing platform that runs the whole system in a controlled environment and can replay any moment of a run exactly (very cool!). I had a working local setup, but for some reason the engine pod was not getting discovered

```text
checkIfWeCanExecute Q731695-...: query can not proceed to find an engine
  based on in progress count. numQueriesInProgress: 0, maxNumberOfConcurrentQueries: 0
```

Alongside it, the component that opens a connection to the executor kept failing:

```text
ERROR NodeManager ... Trying to connect to: 10.85.0.6, java.net.ConnectException: Connection refused
```

So the node-manager could see the engine pod and kept trying to reach it, but every connection was refused. Connection refused only means that nothing is listening on that port. It does not say whether the engine had crashed, was running but slow, or was stuck before it opened the port. So this became the first thing to find out.

The cheapest explanation is that the problem is in our own code or in a specific query. But the exact same build had passed our test-suite on a local cluster (on both Arm and x86), right before I sent it to Antithesis. The binary and the tests were identical, so whatever was happening was specific to this environment, not to our code or a particular test.

Because Antithesis records a run deterministically, I could re-enter the exact state of the engine at the moment the node-manager first failed and inspect it with the usual Linux toolbox. Everything below was collected that way.

## What did k8s think of this?

```console
NAMESPACE   NAME                                   READY   STATUS    RESTARTS   AGE
default     e6-engine-8647776d9c-gt5f9             1/1     Running   0          102s
...
```

Kubernetes had not restarted the pod, and the node-manager kept selecting it as a valid engine pod to talk to.

So the engine pod

1. has not restarted
2. is running and ready according to k8s
3. and somehow is still not accepting any requests on that port

What kind of a process does this indicate?

1. Kubernetes treats a container as running as long as its main process has started and has not exited.
2. If no probe checks the service port, the pod is ready as soon as the container is running. Whether the pod is also marked ready depends on its readiness probe. If you don't have a probe configured, by default as soon as the process starts the pod is assumed to be ready. What really was the condition for the node-manager to reach out to the engine pod? That k8s marked it as ready.

Some quick debugging and turns out I did a rookie 101 mistake

```console
readinessProbe:   (none)
livenessProbe:    (none)
Ready=True   ContainersReady=True
containerReady: true  started=true  state={"running":{"startedAt":"2026-04-17T14:03:06Z"}}
```

I forgot to enable the readiness probe. So the pod wasn't really _ready_ and led to k8s thinking otherwise. So why wasn't it booting up?

Was it _stuck_?

## Prereqs

During my undergrad, I had worked on using a [syscall interceptor](https://github.com/pmem/syscall_intercept) for an file system project. So I was somewhat familiar with the usual suspects of these situations. Some of the useful things that will be required to understand the analysis below -

* The `/proc/<pid>/` directory is a live view of a running process that the kernel provides. The files under it report things like the number of threads, the memory in use, the system call currently in progress, and how many bytes have been written to each open file.
* A futex is the low-level mechanism that most locks are built on. When a thread cannot take a lock, it goes to sleep inside a `futex` system call. One of the arguments to that call is a timeout. If the timeout is empty, the thread waits with no time limit.
* A program does not _really_ begin at `main`. Before `main` runs, the dynamic linker runs a list of initialization functions called constructors. Code in a constructor runs while the process is only partly set up.
* A backtrace is the list of function calls that led to the current point in the program. It is read from the top, where frame 0 is the function running right now. When debuggers (like gdb) cannot find the names for a library, it prints the frame as `??`.
* Our engine uses jemalloc as its memory allocator, so every call to `malloc` in the program goes through jemalloc.

## First question: is the process even alive?

Before anything else I wanted to know whether the executor process existed and what state it was in. If it had crashed there would be nothing to inspect.

```console
$ ps -eo pid,nlwp,stat,comm | grep engine
2830  1  Ss  e6-engine
```

The process was there, with process id 2830. The `stat` column showed `Ss`. The first letter, `S`, means the process is sleeping, not dead and not a zombie. So the engine was alive and simply asleep. It had not crashed or anything. So this was no crash investigation (cues in NatGeo's Air Crash Investigation theme). Something was making the process wait.

## Looking at the one thread

The `nlwp` column in the earlier `ps` output is the number of threads, and it was one. Our engine uses an async runtime that normally starts many worker threads once it is up, so a single thread meant it had not gotten far enough in startup to create any of them. Whatever the process was doing, it was doing it very early. That was an important clue. To get the answer I looked at what the one thread was actually doing.

```console
$ cat /proc/2830/wchan; echo
futex_wait_queue
```

The `wchan` file stands for "wait channel." When a process is asleep inside the kernel, this file gives the name of the kernel function it is blocked in. Here it was `futex_wait_queue`, which is the kernel code that a thread sits in while it waits on a futex. So the thread was not doing work at all. It was parked, waiting for a lock.

That raised the next question: waiting on which lock, and for how long? The `syscall` file shows the system call currently in progress and its arguments.

```console
$ cat /proc/2830/syscall
202 0x55aac6a269e0 0x80 0x2 0x0 0x0 0x0 0x7ffe3e70bf18 0x7f2c8aeb912b
```

The first number is the system call number. Number 202 on x86-64 is [`futex`](https://man7.org/linux/man-pages/man2/futex.2.html) and the number-to-name mapping for this architecture is the kernel's [syscall table](https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl), where line 202 reads `futex`. The remaining numbers are the arguments to that call, and they are the useful part:

* The first argument,  `0x55aac6a269e0`, is the address of the lock it is waiting on. No use here
* The second argument,  `0x80`, means it is a normal wait on a private lock.
* The third argument,  `0x2`, is the value it expects the lock to have. For a standard lock, the value 2 means the lock is held and at least one thread is already waiting.
* The fourth argument,  `0x0`, is the timeout, and it is empty. The thread will wait with no time limit.

The empty timeout is the important one. This thread would wait forever unless something released the lock. A confirmation check that the thread was genuinely parked and not briefly passing through:

```console
$ grep -E 'VmRSS|^voluntary' /proc/2830/status
VmRSS:   16896 kB
voluntary_ctxt_switches:  244
```

It was using about 16MB of memory, which is very little, and its context switch count did not change when I sampled it again. So this was one thread, asleep on a lock with no time limit, that had done almost no work.

Now the single-thread fact mattered. A lock is released by whichever thread holds it. There was only one thread, and it was asleep waiting for the lock, so nothing was left to release it. The process was not going to recover on its own.

_Single thread deadlocked?_

## Confirming it never reached `main`

The engine pod was showing no logs whatever, even the log lines and system config that we print as the first thing in main. Was it possible that it was writing the logs, but we weren't catching it somehow?

To check this, I inspected whether the process had ever written any output. File descriptors 1 and 2 are standard output and standard error, and the `fdinfo` file reports how many bytes have passed through each one.

```console
$ readlink /proc/2830/fd/1
pipe:[11707]
$ grep ^pos /proc/2830/fdinfo/1
pos: 0
$ readlink /proc/2830/fd/2
pipe:[11708]
$ grep ^pos /proc/2830/fdinfo/2
pos: 0
```

Both read `pos: 0`, so I assumed the process had never written a single byte to either stream. (And here's a catch, this is technically wrong. `pos` is the file cursor's offset and it only tracks bytes for a seekable file. Pipes do not have any cursor since the bytes are consumed and discarded as the reader drains them. To the best of my knowledge, there is no inbuilt tracker for how many bytes flowed through the pipe. A check using `strace` would have been more appropriate. Fortunately, the conclusion was still correct, albeit via an faulty reasoning approach)

Second, I checked whether it had ever opened its network port, since the node-manager was getting connection refused on 9002.

```console
$ grep ' 0A ' /proc/2830/net/tcp
$
```

0A is hex for 10, which is TCP_LISTEN in the kernel's [socket-state enum](https://github.com/torvalds/linux/blob/master/include/net/tcp_states.h) (thanks to the Network Programming course in my undergrad that I still remembered where to find this). So this also eliminated that the hypothesis that connection-refused errors was a netowrk fault, there was simply nothing listening here at all.

So the process was stuck during startup, before it wrote any output and before it opened its port, waiting on a lock that would never be released. The only code that runs that early is the constructors that run before `main`. To see which one, I needed a backtrace, and the `/proc` files do not provide that. So I moved to a debugger.

## Shenanigans with GDB

The debugger, gdb, was available on the host, so I attached to the process directly.

```console
$ gdb -p 2830 -batch -ex bt
warning: Target and debugger are in different PID namespaces; thread lists and
         other data are likely unreliable.
warning: "target:/proc/2830/exe": could not open as an executable file: Input/output error.
#0  0x00007f2c8aeb912b in ?? () from .../libc.so.6
#1  0x00007f2c8aebf482 in pthread_mutex_lock () from .../libc.so.6
#2  0x000055aac13cc572 in ?? ()
#3  0x0000000000000000 in ?? ()
```

This did not tell me much. The engine runs inside a container, which has its own process namespace, and gdb was running on the host in a different namespace. Because of that, gdb could not even read the engine's own program file, and the backtrace stopped after two frames. The one useful line was frame 1, which confirmed the thread was inside `pthread_mutex_lock`, but I already knew that from the futex. Two frames were not enough to see which code took the lock.

GDB does the range mapping for each of the frames in the output when the `bt` flag is passed. Every stack frame has an associated program counter. `/proc/<pid>/maps` lists every mapped region of the process, so the attribution procedure was just: take the program counter address, find which `maps` range contains it, and then check the file containing that range. The file is the library the frame was running in.

For instance, a `grep` on `/proc/2830/maps` with our search keyterms `gcc`, `libgcc`, etc. would give this as the output
```console
55aabef6d000-55aac0daa000 r--p 00000000 07:01 85504                      /app/e6-engine
55aac0daa000-55aac6564000 r-xp 01e3c000 07:01 85504                      /app/e6-engine
55aac6564000-55aac6a0f000 r--p 075f5000 07:01 85504                      /app/e6-engine
55aac6a0f000-55aac6a27000 rw-p 07a9f000 07:01 85504                      /app/e6-engine
7f2c8ae33000-7f2c8ae59000 r--p 00000000 07:01 5905                       /usr/lib/x86_64-linux-gnu/libc.so.6
7f2c8ae59000-7f2c8afaf000 r-xp 00026000 07:01 5905                       /usr/lib/x86_64-linux-gnu/libc.so.6
7f2c8afaf000-7f2c8b002000 r--p 0017c000 07:01 5905                       /usr/lib/x86_64-linux-gnu/libc.so.6
7f2c8b002000-7f2c8b006000 r--p 001cf000 07:01 5905                       /usr/lib/x86_64-linux-gnu/libc.so.6
7f2c8b006000-7f2c8b008000 rw-p 001d3000 07:01 5905                       /usr/lib/x86_64-linux-gnu/libc.so.6
7f2c8b0f7000-7f2c8b0fa000 r--p 00000000 07:01 5927                       /usr/lib/x86_64-linux-gnu/libgcc_s.so.1
7f2c8b0fa000-7f2c8b111000 r-xp 00003000 07:01 5927                       /usr/lib/x86_64-linux-gnu/libgcc_s.so.1
7f2c8b111000-7f2c8b115000 r--p 0001a000 07:01 5927                       /usr/lib/x86_64-linux-gnu/libgcc_s.so.1
7f2c8b115000-7f2c8b116000 r--p 0001d000 07:01 5927                       /usr/lib/x86_64-linux-gnu/libgcc_s.so.1
7f2c8b116000-7f2c8b117000 rw-p 0001e000 07:01 5927                       /usr/lib/x86_64-linux-gnu/libgcc_s.so.1
7f2c8b64a000-7f2c8b6a6000 r--p 00000000 07:01 137052                     /usr/lib/libvoidstar.so
7f2c8b6a6000-7f2c8b763000 r-xp 0005c000 07:01 137052                     /usr/lib/libvoidstar.so
7f2c8b763000-7f2c8b76b000 r--p 00119000 07:01 137052                     /usr/lib/libvoidstar.so
7f2c8b76b000-7f2c8b76c000 rw-p 00121000 07:01 137052                     /usr/lib/libvoidstar.so
7f2c8b770000-7f2c8b771000 r--p 00000000 07:01 5887                       /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
7f2c8b771000-7f2c8b797000 r-xp 00001000 07:01 5887                       /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
7f2c8b797000-7f2c8b7a1000 r--p 00027000 07:01 5887                       /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
7f2c8b7a1000-7f2c8b7a3000 r--p 00031000 07:01 5887                       /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
7f2c8b7a3000-7f2c8b7a5000 rw-p 00033000 07:01 5887                       /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
```

The fix was to tell gdb exactly where the files were instead of letting it guess. The container's files are visible from the host under `/proc/2830/root`. I pointed gdb at the executor binary there, set that directory as the root for finding shared libraries, and then attached.

```console
$ gdb -batch -nx \
    -ex 'set sysroot /proc/2830/root' \
    -ex 'file /proc/2830/root/app/e6-engine' \
    -ex 'attach 2830' \
    -ex 'bt'
```

Here's the output with the attribution:

```console
#0  0x00007f2c8aeb912b in ?? () from /proc/2830/root/lib/x86_64-linux-gnu/libc.so.6
#1  0x00007f2c8aebf482 in pthread_mutex_lock () from /proc/2830/root/lib/x86_64-linux-gnu/libc.so.6
#2  0x000055aac13cc572 in _rjem_je_malloc_mutex_lock_slow ()
#3  0x000055aac135e459 in malloc_init_hard ()
#4  0x000055aac13638cd in _rjem_je_malloc_default ()
#5  0x00007f2c8b10f73d in ?? () from /proc/2830/root/lib/x86_64-linux-gnu/libgcc_s.so.1
#6  0x00007f2c8b110046 in _Unwind_Find_FDE () from /proc/2830/root/lib/x86_64-linux-gnu/libgcc_s.so.1
#7  0x00007f2c8b10c503 in ?? () from /proc/2830/root/lib/x86_64-linux-gnu/libgcc_s.so.1
#8  0x00007f2c8b10d7a0 in ?? () from /proc/2830/root/lib/x86_64-linux-gnu/libgcc_s.so.1
#9  0x00007f2c8b10e5cc in _Unwind_Backtrace () from /proc/2830/root/lib/x86_64-linux-gnu/libgcc_s.so.1
#10 0x000055aac13d0ed7 in _rjem_je_prof_boot2 ()
#11 0x000055aac135e3f2 in malloc_init_hard ()
#12 0x000055aac13638cd in _rjem_je_malloc_default ()
#13 0x00007f2c8b72269a in ?? () from /proc/2830/root/usr/lib/libvoidstar.so
#14 0x00007f2c8b75ed0b in ?? () from /proc/2830/root/usr/lib/libvoidstar.so
#15 0x00007f2c8b7749ce in ?? ()
#16 0x0000000000000000 in ?? ()
```

## What the backtrace says

Reading it from the bottom up gives the whole sequence.

Frame 15 does not have any attribution, but cross referencing it in the list of addresses in `/maps` shows it in the range of `ld-linux`'s executable segment. So we know it is the dynamic linker.

The dynamic linker at frame 15 is running constructors before `main`. One of those constructors is in libvoidstar, at frames 13 and 14. This is a library that Antithesis adds to the program to record [code coverage](https://antithesis.com/docs/configuration/the_antithesis_environment/#library-path). That constructor calls `malloc`.

That call at frame 12 is the first `malloc` in the entire process. Because our allocator is jemalloc, the first call triggers jemalloc's one-time setup, `malloc_init_hard` at frame 11, which takes an internal lock while it runs. I confirmed that the lock the thread is stuck on is exactly this one:

```console
$ gdb -batch -nx \
  -ex 'set sysroot /proc/2830/root' \
  -ex 'file /proc/2830/root/app/e6-executor' \
  -ex 'attach 2830' \
  -ex 'info symbol 0x55aac6a269e0' \
  -ex 'detach'
init_lock + 64 in section .bss of /app/e6-engine
```

The address from the futex is jemalloc's `init_lock`.

While still inside that setup, jemalloc runs `prof_boot2` at frame 10, which prepares its heap profiler. As part of that preparation it takes a single backtrace of the current stack, which calls into the system's stack-unwinding library, libgcc, at frames 9 down to 6. The first time that library unwinds a stack, it builds an internal cache, and to build that cache it calls `malloc`, at frame 4.

That second `malloc` at frame 3 enters `malloc_init_hard` again. But `malloc_init_hard` is already running further down the stack at frame 11 and is still holding `init_lock`. So the same thread tries to take a lock it is already holding. The lock does not allow that, so the thread goes to sleep waiting for a lock that it will never release, because it is the one holding it. There is no other thread to release it. The process never finishes startup, never reaches `main`, and never opens its port.

At this point I looked at the jemalloc settings that were compiled into the binary.

```console
$ grep -a 'prof:' /app/e6-engine
prof:true,prof_active:false,lg_prof_sample:19
```

My first reaction was that this looked fine, because it said `prof_active:false`. The idea was to include the profiler in the build but leave it turned off, so that an operator could turn it on later if they ever needed a memory profile. If the profiler was off, I did not understand why its setup code was running at all.

And this was a mistake. `prof_active:false` does not mean the profiler is not set up. It means the profiler is set up but not yet collecting samples. The setting that decides whether `prof_boot2` runs its backtrace during startup is `prof:true`, not `prof_active`. The profiler option was on. `prof:true` turned it on, and `prof_active:false` had nothing to do with the startup step.

## Why this only happened on Antithesis

The obvious question is why the same binary ran fine on our local cluster but hung on Antithesis. The difference is libvoidstar, the coverage library that Antithesis adds. Its constructor allocates memory before `main`, which makes it the first `malloc` in the process, earlier than anything in our own startup. That ordering is what closes the loop, because the whole chain only forms when the profiler's backtrace runs during the very first call to `malloc_init_hard`, while the lock is held.

On the local cluster there is no libvoidstar, so some earlier allocation runs first, and by the time the profiler takes its backtrace the unwinding library has already built its cache and does not need to call `malloc`. Without that second `malloc`, the loop never forms. However, I did not fully trace the exact earlier path that keeps the cache warm on the local cluster, but the fix doesn't change.

This is a known problem in jemalloc, not something unique to us. The same sequence, where the profiler setup takes a backtrace during initialization that calls back into `malloc`, has been reported several times:

* [jemalloc issue #585, "deadlock in je_prof_boot2"](https://github.com/jemalloc/jemalloc/issues/585)
* [jemalloc issue #2282, "deadlock when MALLOC_CONF=prof:true"](https://github.com/jemalloc/jemalloc/issues/2282)
* [jemalloc issue #2454, "memory profiling with static executable crashes or deadlocks"](https://github.com/jemalloc/jemalloc/issues/2454): this one shows almost exactly our stack, `malloc_init_hard` calling `prof_boot2` calling `_Unwind_Backtrace`, and the reporter describes the same loop: the unwinder init calls `malloc`, which tries to initialize again.
* [Debian bug #918742, "Initialization loop/deadlock when used with jemalloc"](https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=918742)

The new part in our case was that an Antithesis coverage library was the thing that ran the first allocation. One caveat on the mechanism: some jemalloc versions run the unwinder probe during init even with the profiler off, so disabling it is not guaranteed to help everywhere.

## The fix

The fix was to not turn the profiler on during startup. I set the jemalloc configuration to an empty value:

```toml
malloc_conf = "\0"
```

With this, the profiler option is off when `malloc_init_hard` runs. `prof_boot2` returns early, does not take a backtrace, and never triggers the second `malloc`. The first `malloc_init_hard` finishes normally, releases its lock, the libvoidstar constructor returns, the linker continues, and `main` runs.

The profiler is still compiled into the binary. There is one catch. Because the problem is caused by the profiler being on at startup, the same binary cannot both leave the profiler armed and be safe under libvoidstar. Turning the profiler back on with an environment variable brings the deadlock back. So the profiler stays off in the Antithesis build, and enabled everywhere else.

## What I would tell myself next time

* Assuming failure modes before having any evidence is always a bad idea. When I started debugging this, I had no clue whether the process had crashed or was slow or was stuck. In hindsight, the first useful thing I did was to check whether the process was alive and sleeping, which ruled out a crash before I spent any time on it.
* `/proc` files are a godsend. It can answer a lot of the questions before we even had to reach to the debugger. Things like process state, thread count, the waiting channel, what is the current system call, what are its arguments, what are its output streams, what else the sockets it is listening on answered a lot of the questions.
* A single thread asleep on a lock is a strong signal. If there is only one thread and it is waiting for a lock, there is nothing left that can release the lock, so the process will not recover by itself.
* When gdb gives `??` frames across a container boundary, point it at the binary and set the root directory yourself instead of attaching by process id alone.
* Read your own configuration carefully. The `prof_active:false` setting cost me more time than the backtrace did, because I trusted what I thought it meant instead of checking what it actually controlled.
