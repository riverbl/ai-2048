#![feature(
    control_flow_enum,
    int_roundings,
    iter_array_chunks,
    maybe_uninit_slice,
    maybe_uninit_uninit_array,
    stmt_expr_attributes,
    trusted_len,
    write_all_vectored
)]

use std::{
    cmp, env,
    io::{self, Read, Write},
    ops::ControlFlow,
    os::fd::AsRawFd,
};

use ai::Ai;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::ai::{expectimax::ExpectimaxAi, monte_carlo::MonteCarloAi, random::RandomAi};

mod ai;
mod control_flow_helper;
mod direction;
mod logic;
mod render;
mod rng_seeds;

fn play_interactive(
    out: &mut (impl AsRawFd + Write),
    input: &mut impl Read,
    mut rng: impl Rng,
) -> io::Result<()> {
    let mut board = logic::spawn_square(&mut rng, 0);
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
    let mut board = logic::spawn_square(&mut rng, 0);
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

fn bench_ai<A, R, I>(out: &mut impl Write, init_iter: I) -> io::Result<()>
where
    A: Ai,
    R: Rng,
    I: IntoIterator<Item = (A, R)>,
{
    struct Stats {
        runs: u32,
        max_turns: u32,
        max_score: u32,
        min_turns: u32,
        min_score: u32,
        avg_turns: f64,
        avg_score: f64,
    }

    let mut bench_results = init_iter.into_iter().map(|(mut ai, mut rng)| {
        let board = logic::spawn_square(&mut rng, 0);

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

            let avg_turns = stats
                .avg_turns
                .mul_add(f64::from(stats.runs), f64::from(turns))
                / f64::from(runs);

            let avg_score = stats
                .avg_score
                .mul_add(f64::from(stats.runs), f64::from(score))
                / f64::from(runs);

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

fn main() -> io::Result<()> {
    enum Mode {
        Interactive,
        Expectimax(u32),
        MonteCarlo(u32),
        Random,
        BenchExpectimax(u32),
        BenchMonteCarlo(u32),
        BenchRandom,
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
        _ => return writeln!(stdout, "Invalid arguments"),
    };

    match mode {
        Mode::Interactive => play_interactive(&mut stdout, &mut stdin, ChaCha8Rng::from_entropy()),
        Mode::Expectimax(depth) => play_ai(
            &mut stdout,
            ExpectimaxAi::new(depth),
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
            let init_iter = rng_seeds::SEEDS
                .into_iter()
                .step_by(2)
                .map(ChaCha8Rng::from_seed)
                .map(|rng| (ExpectimaxAi::new(depth), rng));

            bench_ai(&mut stdout, init_iter)
        }
        Mode::BenchMonteCarlo(iterations) => {
            let init_iter = rng_seeds::SEEDS
                .into_iter()
                .map(ChaCha8Rng::from_seed)
                .array_chunks()
                .map(|[game_rng, ai_rng]| (MonteCarloAi::new(ai_rng, iterations), game_rng));

            bench_ai(&mut stdout, init_iter)
        }
        Mode::BenchRandom => {
            let init_iter = rng_seeds::SEEDS
                .into_iter()
                .map(ChaCha8Rng::from_seed)
                .array_chunks()
                .map(|[game_rng, ai_rng]| (RandomAi::new(ai_rng), game_rng));

            bench_ai(&mut stdout, init_iter)
        }
    }
}
