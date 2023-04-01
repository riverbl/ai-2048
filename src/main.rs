#![feature(
    control_flow_enum,
    int_roundings,
    iter_array_chunks,
    maybe_uninit_slice,
    maybe_uninit_uninit_array,
    trusted_len,
    write_all_vectored
)]

use std::{
    cmp, env,
    io::{self, Read, Write},
    ops::ControlFlow,
    os::fd::AsRawFd,
};

// use clap::value_parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod ai;
mod control_flow_helper;
mod direction;
mod logic;
mod render;
mod rng_seeds;

fn play_interactive(
    out: &mut (impl AsRawFd + Write),
    input: &mut impl Read,
    rng: &mut impl Rng,
) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
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
                let new_board = logic::spawn_square(rng, new_board);

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

fn play_ai(out: &mut (impl AsRawFd + Write), rng: &mut impl Rng, depth: u32) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    let mut ai = ai::Ai::new(ChaCha8Rng::from_entropy(), depth, 0);

    while move_boards.iter().any(Option::is_some) {
        let direction = ai.get_next_move_expectimax(board).unwrap();

        let new_board = move_boards[direction as usize].unwrap().get();
        let move_score = logic::eval_score(new_board) - logic::eval_score(board);
        let new_board = logic::spawn_square(rng, new_board);

        render::redraw_board(out, board, new_board, score, score + move_score)?;

        board = new_board;
        score += move_score;

        move_boards = logic::try_all_moves(board);
    }

    out.write_all(b"Game over\n")
}

fn play_monte_carlo(
    out: &mut (impl AsRawFd + Write),
    rng: &mut impl Rng,
    iterations: u32,
) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    let mut ai = ai::Ai::new(ChaCha8Rng::from_entropy(), 0, iterations);

    while move_boards.iter().any(Option::is_some) {
        let direction = ai.get_next_move_monte_carlo(board).unwrap();

        let new_board = move_boards[direction as usize].unwrap().get();
        let move_score = logic::eval_score(new_board) - logic::eval_score(board);
        let new_board = logic::spawn_square(rng, new_board);

        render::redraw_board(out, board, new_board, score, score + move_score)?;

        board = new_board;
        score += move_score;

        move_boards = logic::try_all_moves(board);
    }

    out.write_all(b"Game over\n")
}

fn bench(out: &mut impl Write, param: u32) -> io::Result<()> {
    struct Stats {
        runs: u32,
        max_turns: u32,
        max_score: u32,
        min_turns: u32,
        min_score: u32,
        avg_turns: f64,
        avg_score: f64,
    }

    let mut bench_results = rng_seeds::SEEDS
        .into_iter()
        .array_chunks()
        .map(|[seed1, seed2]| {
            let mut game_rng = ChaCha8Rng::from_seed(seed1);
            let ai_rng = ChaCha8Rng::from_seed(seed2);

            let board = logic::spawn_square(&mut game_rng, 0);
            let mut ai = ai::Ai::new(ai_rng, param, param);

            (0..)
                .try_fold((0, 0, board), |(turns, score, board), _| {
                    let maybe_direction = ai.get_next_move_monte_carlo(board);

                    if let Some(direction) = maybe_direction {
                        let new_board = logic::try_move(board, direction).unwrap().get();
                        let move_score = logic::eval_score(new_board) - logic::eval_score(board);
                        let new_board = logic::spawn_square(&mut game_rng, new_board);

                        ControlFlow::Continue((turns + 1, score + move_score, new_board))
                    } else {
                        ControlFlow::Break((turns, score))
                    }
                })
                .break_value()
                .unwrap()
        });

    if let Some((first_turns, first_score)) = bench_results.next() {
        let init_stats = Stats {
            runs: 1,
            max_turns: first_turns,
            max_score: first_score,
            min_turns: first_turns,
            min_score: first_score,
            avg_turns: f64::from(first_turns),
            avg_score: f64::from(first_score),
        };

        let Stats {
            runs,
            max_turns,
            max_score,
            min_turns,
            min_score,
            avg_turns,
            avg_score,
        } = bench_results.fold(init_stats, |stats, (turns, score)| {
            let runs = stats.runs + 1;

            let max_turns = cmp::max(turns, stats.max_turns);
            let max_score = cmp::max(score, stats.max_score);

            let min_turns = cmp::min(turns, stats.min_turns);
            let min_score = cmp::min(score, stats.min_score);

            let avg_turns =
                (stats.avg_turns * f64::from(stats.runs) + f64::from(turns)) / f64::from(runs);
            let avg_score =
                (stats.avg_score * f64::from(stats.runs) + f64::from(score)) / f64::from(runs);

            Stats {
                runs,
                max_turns,
                max_score,
                min_turns,
                min_score,
                avg_turns,
                avg_score,
            }
        });

        write!(
            out,
            "Played {runs} games:\n\
        Max turns {max_turns}, max score {max_score}\n\
        Min turns {min_turns}, min score {min_score}\n\
        Average turns {avg_turns}, average score {avg_score}\n"
        )
    } else {
        writeln!(out, "Empty initialisation iterator")
    }
}

fn play_random(out: &mut (impl AsRawFd + Write), rng: &mut impl Rng) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    while move_boards.iter().any(Option::is_some) {
        let direction = logic::get_next_move_random(rng, board).unwrap();

        let new_board = move_boards[direction as usize].unwrap().get();
        let move_score = logic::eval_score(new_board) - logic::eval_score(board);
        let new_board = logic::spawn_square(rng, new_board);

        render::redraw_board(out, board, new_board, score, score + move_score)?;

        board = new_board;
        score += move_score;

        move_boards = logic::try_all_moves(board);
    }

    out.write_all(b"Game over\n")
}

fn main() -> io::Result<()> {
    enum Mode {
        Interactive,
        Ai(u32),
        MonteCarlo(u32),
        Random,
        Bench(u32),
    }

    // let arg_parser = clap::Command::new("ai2048")
    //     .arg(
    //         clap::Arg::new("Random")
    //             .short('r')
    //             .long("rand")
    //             .conflicts_with_all(["AI", "Depth"]),
    //     )
    //     .arg(
    //         clap::Arg::new("AI")
    //             .short('a')
    //             .long("--ai")
    //             .requires("Depth"),
    //     )
    //     .arg(
    //         clap::Arg::new("Depth")
    //             .short('d')
    //             .long("--depth")
    //             .requires("AI")
    //             .value_parser(value_parser!(u32)),
    //     );

    let mut stdout = io::stdout().lock();
    let mut stdin = io::stdin().lock();

    // let args = match arg_parser.try_get_matches_from(env::args_os()) {
    //     Ok(args) => args,
    //     Err(err) => return write!(stdout, "{err}"),
    // };

    let args: Vec<_> = env::args().skip(1).collect();

    let mode = match args.as_slice() {
        [] => Mode::Interactive,
        [arg, depth_str] if arg == "-a" => {
            let Ok(depth) = depth_str.parse() else {
                return writeln!(stdout, "Invalid depth {depth_str}");
            };

            Mode::Ai(depth)
        }
        [arg, iterations_str] if arg == "-m" => {
            let Ok(iterations) = iterations_str.parse() else {
                return writeln!(stdout, "Invalid iterations {iterations_str}");
            };

            Mode::MonteCarlo(iterations)
        }
        [arg] if arg == "-r" => Mode::Random,
        [arg, param_str] if arg == "-b" => {
            let Ok(param) = param_str.parse() else {
                return writeln!(stdout, "Invalid param {param_str}");
            };

            Mode::Bench(param)
        }
        _ => return writeln!(stdout, "Invalid arguments"),
    };

    let mut rng = ChaCha8Rng::from_entropy();

    match mode {
        Mode::Interactive => play_interactive(&mut stdout, &mut stdin, &mut rng),
        Mode::Ai(depth) => play_ai(&mut stdout, &mut rng, depth),
        Mode::MonteCarlo(iterations) => play_monte_carlo(&mut stdout, &mut rng, iterations),
        Mode::Random => play_random(&mut stdout, &mut rng),
        Mode::Bench(param) => bench(&mut stdout, param),
    }
}
