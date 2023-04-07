use std::num::NonZeroU64;

use crate::{direction::Direction, logic};

pub mod expectimax;
pub mod monte_carlo;
pub mod random;

pub trait Ai {
    fn get_next_move(&mut self, board: u64) -> Option<Direction>;
}

const MOVE_FUNCTIONS: [fn(u64) -> Option<NonZeroU64>; 4] =
    [move_up, move_down, move_right, move_left];

fn move_up(board: u64) -> Option<NonZeroU64> {
    let board = logic::transpose_board(board);

    let new_board = logic::do_move(board);

    (new_board != board)
        .then_some(NonZeroU64::new(new_board))
        .flatten()
}

fn move_down(board: u64) -> Option<NonZeroU64> {
    let board = logic::transpose_rotate_board(board);

    let new_board = logic::do_move(board);

    (new_board != board)
        .then_some(NonZeroU64::new(new_board))
        .flatten()
}

fn move_right(board: u64) -> Option<NonZeroU64> {
    let board = logic::mirror_board(board);

    let new_board = logic::do_move(board);

    (new_board != board)
        .then_some(NonZeroU64::new(new_board))
        .flatten()
}

fn move_left(board: u64) -> Option<NonZeroU64> {
    let new_board = logic::do_move(board);

    (new_board != board)
        .then_some(NonZeroU64::new(new_board))
        .flatten()
}

fn try_move(board: u64, direction: Direction) -> Option<NonZeroU64> {
    MOVE_FUNCTIONS[direction as usize](board)
}

fn get_all_moves(board: u64) -> impl Iterator<Item = (u64, Direction)> {
    Direction::iter()
        .filter_map(move |direction| {
            try_move(board, direction).map(|new_board| (new_board, direction))
        })
        .map(|(board, direction)| (board.get(), direction))
}
