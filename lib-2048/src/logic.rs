use std::{arch::x86_64::_pext_u64, mem, num::NonZeroU64};

use rand::Rng;

use crate::direction::Direction;

static MOVE_TABLE: [u16; 1 << 16] = {
    static TABLE: &[u8; 1 << 17] = include_bytes!(concat!(env!("OUT_DIR"), "/move_table"));

    unsafe { mem::transmute_copy(TABLE) }
};
static SCORE_TABLE: [u32; 1 << 8] = {
    static TABLE: &[u8; 1 << 10] = include_bytes!(concat!(env!("OUT_DIR"), "/score_table"));

    unsafe { mem::transmute_copy(TABLE) }
};
static TURN_COUNT_TABLE: [u16; 1 << 8] = {
    static TABLE: &[u8; 1 << 9] = include_bytes!(concat!(env!("OUT_DIR"), "/turn_count_table"));

    unsafe { mem::transmute_copy(TABLE) }
};
static MID_METRICS_TABLE: [(f32, f32); 1 << 16] = {
    static TABLE: &[u8; 1 << 19] = include_bytes!(concat!(env!("OUT_DIR"), "/mid_metrics_table"));

    unsafe { mem::transmute_copy(TABLE) }
};
static EDGE_METRICS_TABLE: [(f32, f32); 1 << 16] = {
    static TABLE: &[u8; 1 << 19] = include_bytes!(concat!(env!("OUT_DIR"), "/edge_metrics_table"));

    unsafe { mem::transmute_copy(TABLE) }
};

const MOVE_FUNCTIONS: [fn(u64) -> u64; 4] = [move_up, move_down, move_right, move_left];

pub fn get_opponent_moves(board: u64) -> impl Iterator<Item = (u64, f64)> {
    let (slot_count, slots) = get_empty_slots(board);

    let two_probability = 0.9 / f64::from(slot_count);
    let four_probability = 0.1 / f64::from(slot_count);

    (0..slot_count).flat_map(move |slot_idx| {
        [(1, two_probability), (2, four_probability)]
            .into_iter()
            .map(move |(cell, probability)| {
                let slot = (slots >> (slot_idx * 4)) & 0xf;
                let new_board = board | (cell << (slot * 4));

                (new_board, probability)
            })
    })
}

/// Returns an integer with the least significant bit of each nibble that is `0` in `board` set to
/// `1` and all other bits set to `0`.
///
/// ```
/// # use lib_2048::logic::mark_empty_cells;
///
/// assert_eq!(mark_empty_cells(0x0032_1000_0a00_0000), 0x1100_0111_1011_1111);
/// ```
pub const fn mark_empty_cells(board: u64) -> u64 {
    let table = board | (board >> 1);
    let table = table | (table >> 2);

    !table & 0x1111_1111_1111_1111
}

// const fn count_empty_cells(board: u64) -> u32 {
//     let empty_cells = mark_empty_cells(board);

//     empty_cells.count_ones()
// }

/// Returns a tuple where the first item is the number of `0` nibbles in `board` and the second
/// item is an integer containing the indices of all `0` nibbles in `board`.
///
/// The second item contains a list of the indices of all `0` nibbles, with each index packed into
/// a single nibble, starting with the least significant nibble of the second item, in order from
/// the least to most significant empty nibble of `board`.
///
/// ```
/// # use lib_2048::logic::get_empty_slots;
///
/// assert_eq!(get_empty_slots(0x0032_1000_0a00_0000), (12, 0x0000_fea9_8754_3210));
/// ```
pub fn get_empty_slots(board: u64) -> (u32, u64) {
    let empty_cells = mark_empty_cells(board);

    let empty_cells_all_ones = {
        let ones = empty_cells | (empty_cells << 1);
        ones | (ones << 2)
    };

    let slots = unsafe { _pext_u64(0xfedc_ba98_7654_3210, empty_cells_all_ones) };

    (empty_cells.count_ones(), slots)
}

pub fn spawn_square(rng: &mut impl Rng, board: u64) -> u64 {
    let (slot_count, slots) = get_empty_slots(board);

    if slot_count > 0 {
        let rand = rng.gen_range(0..(slot_count * 10));

        let slot_idx = rand / 10;
        let cell = if rand % 10 == 0 { 2 } else { 1 };

        let slot = (slots >> (slot_idx * 4)) & 0xf;

        board | (cell << (slot * 4))
    } else {
        board
    }
}

/// Returns an initial board with 2 randomly spawned tiles.
pub fn get_initial_board(rng: &mut impl Rng) -> u64 {
    (0..2).fold(0, |board, _| spawn_square(rng, board))
}

/// Returns the heuristic turn count for `board`.
///
/// The heuristic turn count is 2 greater than the number of turns that would have been taken to
/// reach the current position if only tiles with value 2 had spawned during the game.
/// The reason it is 2 greater is that the game starts off with 2 tiles already spawned.
///
/// # Examples
///
/// ```
/// # use lib_2048::logic::count_turns;
///
/// assert_eq!(count_turns(0x0032_1000_0a00_0000), 519);
/// ```
pub fn count_turns(board: u64) -> u16 {
    (0..8).fold(0, |turn_count, i| {
        let cell_pair = ((board >> (i * 8)) & 0xff) as usize;

        turn_count + TURN_COUNT_TABLE[cell_pair]
    })
}

/// Returns the heuristic score for `board`.
///
/// The heuristic score is the score that a game in the current position would have if only tiles
/// with value 2 had spawned during the game.
///
/// # Examples
///
/// ```
/// # use lib_2048::logic::eval_score;
///
/// assert_eq!(eval_score(0x0032_1000_0a00_0000), 9_236);
/// ```
pub fn eval_score(board: u64) -> u32 {
    (0..8).fold(0, |score, i| {
        let cell_pair = ((board >> (i * 8)) & 0xff) as usize;

        score + SCORE_TABLE[cell_pair]
    })
}

/// Returns a value indicating how 'good' the position represented by `board` is (higher is
/// better).
///
/// The expectimax AI makes moves that are likely to result in a 'good' position, as determined by
/// this function.
pub fn eval_metrics(board: u64) -> f64 {
    // 1 / scale is expected to grow with number of turns roughly as fast as score.
    let scale = {
        let turn_count: f64 = count_turns(board).into();
        1.0 / (f64::ln(turn_count) * turn_count)
    };

    [board, crate::transpose_board(board)]
        .into_iter()
        .fold(0.0, |total_metrics, board| {
            let mid_metrics: f64 = [16, 32]
                .into_iter()
                .map(|i| {
                    let row = (board >> i) & 0xffff;
                    let (score_metrics, count_metrics) = MID_METRICS_TABLE[row as usize];

                    f64::from(score_metrics).mul_add(scale, count_metrics.into())
                })
                .sum();

            let edge_metrics: f64 = [0, 48]
                .into_iter()
                .map(|i| {
                    let row = (board >> i) & 0xffff;
                    let (score_metrics, count_metrics) = EDGE_METRICS_TABLE[row as usize];

                    f64::from(score_metrics).mul_add(scale, count_metrics.into())
                })
                .sum();

            total_metrics + mid_metrics + edge_metrics
        })
}

pub fn do_move(board: u64) -> u64 {
    (0..4)
        .map(|i| {
            let row = (board >> (i * 16)) as u16;
            (i, MOVE_TABLE[row as usize])
        })
        .fold(0, |new_board, (i, row)| {
            new_board | (u64::from(row) << (i * 16))
        })
}

fn move_up(board: u64) -> u64 {
    let board = crate::transpose_board(board);

    let new_board = do_move(board);

    crate::transpose_board(new_board)
}

fn move_down(board: u64) -> u64 {
    let board = crate::transpose_rotate_board(board);

    let new_board = do_move(board);

    crate::transpose_rotate_board(new_board)
}

fn move_right(board: u64) -> u64 {
    let board = crate::rotate_board(board);

    let new_board = do_move(board);

    crate::rotate_board(new_board)
}

fn move_left(board: u64) -> u64 {
    do_move(board)
}

pub fn try_all_moves(board: u64) -> [Option<NonZeroU64>; 4] {
    MOVE_FUNCTIONS
        .map(|move_fn| move_fn(board))
        .map(|new_board| {
            (new_board != board)
                .then_some(NonZeroU64::new(new_board))
                .flatten()
        })
}

pub fn try_move(board: u64, direction: Direction) -> Option<NonZeroU64> {
    let new_board = MOVE_FUNCTIONS[direction as usize](board);

    (new_board != board)
        .then_some(NonZeroU64::new(new_board))
        .flatten()
}
