#![feature(
    array_windows,
    atomic_bool_fetch_not,
    control_flow_enum,
    int_roundings,
    iter_array_chunks,
    maybe_uninit_array_assume_init,
    stmt_expr_attributes,
    trusted_len,
    write_all_vectored
)]

use std::{
    cmp, env,
    io::{self, Read, Write},
    ops::ControlFlow,
    os::fd::AsRawFd,
    time::Instant,
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use lib_2048::{
    ai::{expectimax::ExpectimaxAi, monte_carlo::MonteCarloAi, random::RandomAi, Ai},
    control_flow_helper, logic, rng_seeds,
};

mod render;

const LOSS_WEIGHT: f64 = -3.9652309010456945;

fn play_interactive(
    out: &mut (impl AsRawFd + Write),
    input: &mut impl Read,
    mut rng: impl Rng,
) -> io::Result<()> {
    let mut board = logic::get_initial_board(&mut rng);
    let mut score = 0;

    let mut buf = [0u8; 128];

    let input_searcher =
        aho_corasick::packed::Searcher::new([b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D"]).unwrap();

    let mut buf_len = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    while move_boards.iter().any(Option::is_some) {
        buf_len += input.read(&mut buf[buf_len..])?;

        for key in input_searcher
            .find_iter(&buf[..buf_len])
            .map(|m| m.pattern())
        {
            if let Some(new_board) = move_boards[key] {
                let new_board = new_board.get();
                let move_score = logic::eval_score(new_board) - logic::eval_score(board);
                let new_board = logic::spawn_square(&mut rng, new_board);

                render::redraw_board(out, board, new_board, score, score + move_score)?;

                board = new_board;
                score += move_score;

                move_boards = logic::try_all_moves(board);
            }
        }

        buf_len = match &buf[..buf_len] {
            [.., 0x1b, b'['] => {
                buf.copy_from_slice(b"\x1b[");
                2
            }
            [.., 0x1b] => {
                buf[0] = 0x1b;
                1
            }
            _ => 0,
        }
    }

    out.write_all(b"Game over\n")
}

fn play_ai(out: &mut (impl AsRawFd + Write), mut ai: impl Ai, mut rng: impl Rng) -> io::Result<()> {
    let mut board = logic::get_initial_board(&mut rng);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    while move_boards.iter().any(Option::is_some) {
        let direction = ai.get_next_move(board).unwrap();

        let new_board = move_boards[direction as usize].unwrap().get();
        let move_score = logic::eval_score(new_board) - logic::eval_score(board);
        let new_board = logic::spawn_square(&mut rng, new_board);

        render::redraw_board(out, board, new_board, score, score + move_score)?;

        board = new_board;
        score += move_score;

        move_boards = logic::try_all_moves(board);
    }

    out.write_all(b"Game over\n")
}

fn bench_ai<W, A, R, I>(out: &mut W, init_iter: I) -> io::Result<()>
where
    W: Write,
    A: Ai,
    R: Rng,
    I: IntoIterator<Item = (A, R)>,
{
    struct Stats {
        run_count: u32,
        max_turns: u32,
        max_score: u32,
        min_turns: u32,
        min_score: u32,
        total_turns: u32,
        total_score: u32,
    }

    let bench_results = init_iter.into_iter().map(|(mut ai, mut rng)| {
        let board = logic::get_initial_board(&mut rng);

        control_flow_helper::loop_try_fold((0, 0, board), |(turns, score, board)| {
            let maybe_direction = ai.get_next_move(board);

            maybe_direction.map_or(ControlFlow::Break((turns, score)), |direction| {
                let new_board = logic::try_move(board, direction).unwrap().get();
                let move_score = logic::eval_score(new_board) - logic::eval_score(board);
                let new_board = logic::spawn_square(&mut rng, new_board);

                ControlFlow::Continue((turns + 1, score + move_score, new_board))
            })
        })
    });

    // Iterators are lazy, so we measure the time taken to iterate the iterator, not the time
    // taken to create it.
    let start = Instant::now();

    let init_stats = Stats {
        run_count: 0,
        max_turns: 0,
        max_score: 0,
        min_turns: u32::MAX,
        min_score: u32::MAX,
        total_turns: 0,
        total_score: 0,
    };

    let Stats {
        run_count,
        max_turns,
        max_score,
        min_turns,
        min_score,
        total_turns,
        total_score,
    } = bench_results.fold(init_stats, |stats, (turns, score)| {
        let run_count = stats.run_count + 1;

        let max_turns = cmp::max(turns, stats.max_turns);
        let max_score = cmp::max(score, stats.max_score);

        let min_turns = cmp::min(turns, stats.min_turns);
        let min_score = cmp::min(score, stats.min_score);

        let total_turns = stats.total_turns + turns;

        let total_score = stats.total_score + score;

        Stats {
            run_count,
            max_turns,
            max_score,
            min_turns,
            min_score,
            total_turns,
            total_score,
        }
    });

    let time_taken = (Instant::now() - start).as_secs_f64();

    if run_count > 0 {
        let avg_turns = f64::from(total_turns) / f64::from(run_count);
        let avg_score = f64::from(total_score) / f64::from(run_count);
        let turns_per_second = f64::from(total_turns) / time_taken;

        write!(
            out,
            "Played {run_count} games:\n\
            Max turns {max_turns}, max score {max_score}\n\
            Min turns {min_turns}, min score {min_score}\n\
            Average turns {avg_turns}, average score {avg_score}\n\
            Time taken {time_taken}\n\
            Turns per second {turns_per_second}\n"
        )
    } else {
        writeln!(out, "Empty initialisation iterator")
    }
}

fn bench_ai_from_seeds<W, R, I, A, F>(out: &mut W, seeds: I, f: F) -> io::Result<()>
where
    W: Write,
    R: Rng + SeedableRng,
    I: IntoIterator<Item = R::Seed>,
    A: Ai,
    F: FnMut([R::Seed; 2]) -> (A, R),
{
    let init_iter = seeds.into_iter().array_chunks().map(f);

    bench_ai(out, init_iter)
}

fn main() -> io::Result<()> {
    enum Mode {
        Interactive,
        Expectimax(u32),
        MonteCarlo(u32),
        Random,
        BenchExpectimax(u32),
        BenchMonteCarlo(u32),
        BenchRandom,
        RandomBenchExpectimax(u32, u32),
    }

    let mut stdout = io::stdout().lock();
    let mut stdin = io::stdin().lock();

    let args: Vec<_> = env::args().skip(1).collect();

    let mode = match args.as_slice() {
        [] => Mode::Interactive,
        [arg, depth_str] if arg == "-e" => {
            let Ok(depth) = depth_str.parse() else {
                return writeln!(stdout, "Invalid depth {depth_str}");
            };

            Mode::Expectimax(depth)
        }
        [arg, iterations_str] if arg == "-m" => {
            let Ok(iterations) = iterations_str.parse() else {
                return writeln!(stdout, "Invalid iterations {iterations_str}");
            };

            Mode::MonteCarlo(iterations)
        }
        [arg] if arg == "-r" => Mode::Random,
        [arg, depth_str] if arg == "--be" => {
            let Ok(depth) = depth_str.parse() else {
                return writeln!(stdout, "Invalid depth {depth_str}");
            };

            Mode::BenchExpectimax(depth)
        }
        [arg, iterations_str] if arg == "--bm" => {
            let Ok(iterations) = iterations_str.parse() else {
                return writeln!(stdout, "Invalid iterations {iterations_str}");
            };

            Mode::BenchMonteCarlo(iterations)
        }
        [arg] if arg == "--br" => Mode::BenchRandom,
        [arg, count_str, depth_str] if arg == "--rbe" => {
            let Ok(count) = count_str.parse() else {
                return writeln!(stdout, "Invalid count {count_str}");
            };

            let Ok(depth) = depth_str.parse() else {
                return writeln!(stdout, "Invalid depth {depth_str}");
            };

            Mode::RandomBenchExpectimax(count, depth)
        }
        _ => return writeln!(stdout, "Invalid arguments"),
    };

    match mode {
        Mode::Interactive => play_interactive(&mut stdout, &mut stdin, ChaCha8Rng::from_entropy()),
        Mode::Expectimax(depth) => play_ai(
            &mut stdout,
            ExpectimaxAi::new(depth, LOSS_WEIGHT, logic::eval_metrics),
            ChaCha8Rng::from_entropy(),
        ),
        Mode::MonteCarlo(iterations) => play_ai(
            &mut stdout,
            MonteCarloAi::new(ChaCha8Rng::from_entropy(), iterations),
            ChaCha8Rng::from_entropy(),
        ),
        Mode::Random => play_ai(
            &mut stdout,
            RandomAi::new(ChaCha8Rng::from_entropy()),
            ChaCha8Rng::from_entropy(),
        ),
        Mode::BenchExpectimax(depth) => {
            bench_ai_from_seeds(&mut stdout, rng_seeds::SEEDS, |[seed, _]| {
                (
                    ExpectimaxAi::new(depth, LOSS_WEIGHT, logic::eval_metrics),
                    ChaCha8Rng::from_seed(seed),
                )
            })
        }
        Mode::BenchMonteCarlo(iterations) => {
            bench_ai_from_seeds(&mut stdout, rng_seeds::SEEDS, |seeds| {
                let [game_rng, ai_rng] = seeds.map(ChaCha8Rng::from_seed);

                (MonteCarloAi::new(ai_rng, iterations), game_rng)
            })
        }
        Mode::BenchRandom => bench_ai_from_seeds(&mut stdout, rng_seeds::SEEDS, |seeds| {
            let [game_rng, ai_rng] = seeds.map(ChaCha8Rng::from_seed);

            (RandomAi::new(ai_rng), game_rng)
        }),
        Mode::RandomBenchExpectimax(count, depth) => {
            let init_iter = (0..count).map(|_| ChaCha8Rng::from_entropy()).map(|rng| {
                (
                    ExpectimaxAi::new(depth, LOSS_WEIGHT, logic::eval_metrics),
                    rng,
                )
            });

            bench_ai(&mut stdout, init_iter)
        }
    }
}
