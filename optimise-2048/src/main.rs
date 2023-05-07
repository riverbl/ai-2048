use std::{
    env,
    io::{self, Write},
    mem::{self, MaybeUninit},
    ptr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

mod optimise;

static INTERRUPT_RECIEVED: AtomicBool = AtomicBool::new(false);

fn set_interrupt_handler() {
    extern "C" fn handle_interrupt(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
        INTERRUPT_RECIEVED.store(true, Ordering::Relaxed);
    }

    let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
        handle_interrupt;

    let sa_mask = unsafe {
        let mut sa_mask = MaybeUninit::uninit();

        libc::sigemptyset(sa_mask.as_mut_ptr());

        sa_mask.assume_init()
    };

    let action = libc::sigaction {
        sa_sigaction: unsafe { mem::transmute(handler) },
        sa_mask,
        sa_flags: libc::SA_RESTART,
        sa_restorer: None,
    };

    unsafe {
        libc::sigaction(libc::SIGINT, &action, ptr::null_mut());
    }
}

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();

    let mut stdout = io::stdout().lock();

    let (per_thread, sigma) = match args.as_slice() {
        [per_thread_str, sigma_str] => {
            let Ok(per_thread) = per_thread_str.parse() else {
            return writeln!(stdout, "Invalid iterations {per_thread_str}");
        };

            let Ok(sigma) = sigma_str.parse() else {
            return writeln!(stdout, "Invalid sigma {per_thread_str}");
        };

            (per_thread, sigma)
        }
        _ => {
            return writeln!(
                stdout,
                "Incorrect argument count: 2 expected, {} provided",
                args.len()
            )
        }
    };

    let thread_count = thread::available_parallelism().unwrap().get();

    set_interrupt_handler();

    optimise::optimise(
        &INTERRUPT_RECIEVED,
        thread_count,
        per_thread,
        sigma,
        &mut stdout,
    )
}
