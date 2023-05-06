use std::{
    env,
    io::{self, Write},
    thread,
};

mod optimise;

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

    optimise::optimise(thread_count, per_thread, sigma, &mut stdout)
}
