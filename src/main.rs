#![feature(int_roundings, never_type, write_all_vectored)]

use std::io::{self, Read, Write};

use rand::{prelude::StdRng, SeedableRng};

mod logic;
mod render;

fn main() -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let mut stdin = io::stdin().lock();

    let mut rng = StdRng::from_entropy();

    let mut board = logic::spawn_square(&mut rng, 0);
    let mut score = 0;

    let mut buf = [0u8; 128];

    let input_searcher =
        aho_corasick::packed::Searcher::new([b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D"]).unwrap();

    let mut buf_len = 0;

    render::setup_terminal(&stdout)?;
    render::draw_board(&mut stdout, board, score)?;

    let mut move_boards = logic::try_all_moves(board);

    while move_boards.iter().any(Option::is_some) {
        buf_len += stdin.read(&mut buf[buf_len..])?;

        for key in input_searcher
            .find_iter(&buf[..buf_len])
            .map(|m| m.pattern())
        {
            if let Some((new_board, move_score)) = move_boards[key] {
                let new_board = logic::spawn_square(&mut rng, new_board);

                render::redraw_board(&mut stdout, board, new_board, score, score + move_score)?;

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

    stdout.write_all(b"Game over\n")
}
