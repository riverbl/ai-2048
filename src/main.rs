#![feature(
    int_roundings,
    maybe_uninit_slice,
    maybe_uninit_uninit_array,
    write_all_vectored
)]

use std::{
    env,
    io::{self, Read, Write},
    os::fd::AsRawFd,
};

// use clap::value_parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod ai;
mod direction;
mod logic;
mod old_ai;
mod render;

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
            if let Some((new_board, move_score)) = move_boards[key] {
                let new_board = logic::spawn_square(rng, new_board.get());

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

fn play_ai(
    out: &mut (impl AsRawFd + Write),
    rng: &mut impl Rng,
    depth: u32,
    iterations: u32,
) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    let mut ai = ai::Ai::new(ChaCha8Rng::from_entropy(), depth, iterations);

    while move_boards.iter().any(Option::is_some) {
        let direction = ai.get_next_move(board).unwrap();

        let (new_board, move_score) = move_boards[direction as usize].unwrap();
        let new_board = logic::spawn_square(rng, new_board.get());

        render::redraw_board(out, board, new_board, score, score + move_score)?;

        board = new_board;
        score += move_score;

        move_boards = logic::try_all_moves(board);
    }

    out.write_all(b"Game over\n")
}

fn play_old_ai(
    out: &mut (impl AsRawFd + Write),
    rng: &mut impl Rng,
    depth: u32,
    iterations: u32,
) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    let mut ai = old_ai::Ai::new(ChaCha8Rng::from_entropy(), depth, iterations);

    while move_boards.iter().any(Option::is_some) {
        let direction = ai.get_next_move(board).unwrap();

        let (new_board, move_score) =
            move_boards[direction as usize].unwrap_or_else(|| panic!("{board}, {direction:?}"));
        let new_board = logic::spawn_square(rng, new_board.get());

        render::redraw_board(out, board, new_board, score, score + move_score)?;

        board = new_board;
        score += move_score;

        move_boards = logic::try_all_moves(board);
    }

    out.write_all(b"Game over\n")
}

fn play_random(out: &mut (impl AsRawFd + Write), rng: &mut impl Rng) -> io::Result<()> {
    let mut board = logic::spawn_square(rng, 0);
    let mut score = 0;

    render::setup_terminal(out)?;
    render::draw_board(out, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    while move_boards.iter().any(Option::is_some) {
        let direction = ai::get_next_move_random(rng, board).unwrap();

        let (new_board, move_score) = move_boards[direction as usize].unwrap();
        let new_board = logic::spawn_square(rng, new_board.get());

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
        Ai(u32, u32),
        OldAi(u32, u32),
        Random,
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
        [arg, depth_str, iterations_str] if arg == "-a" => {
            let Ok(depth) = depth_str.parse::<u32>() else {
                return writeln!(stdout, "Invalid depth {depth_str}");
            };

            let Ok(iterations) = iterations_str.parse::<u32>() else {
                return writeln!(stdout, "Invalid iterations {iterations_str}");
            };

            Mode::Ai(depth, iterations)
        }
        [arg, depth_str, iterations_str] if arg == "-o" => {
            let Ok(depth) = depth_str.parse::<u32>() else {
                return writeln!(stdout, "Invalid depth {depth_str}");
            };

            let Ok(iterations) = iterations_str.parse::<u32>() else {
                return writeln!(stdout, "Invalid iterations {iterations_str}");
            };

            Mode::OldAi(depth, iterations)
        }
        [arg] if arg == "-r" => Mode::Random,
        _ => return writeln!(stdout, "Invalid arguments"),
    };

    let mut rng = ChaCha8Rng::from_entropy();

    match mode {
        Mode::Interactive => play_interactive(&mut stdout, &mut stdin, &mut rng),
        Mode::Ai(depth, iterations) => play_ai(&mut stdout, &mut rng, depth, iterations),
        Mode::OldAi(depth, iterations) => play_old_ai(&mut stdout, &mut rng, depth, iterations),
        Mode::Random => play_random(&mut stdout, &mut rng),
    }
}
