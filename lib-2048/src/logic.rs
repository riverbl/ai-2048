use std::{arch::x86_64::*, mem, num::NonZeroU64};

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
// static METRICS_TABLE: [i8; 1 << 16] = include!(concat!(env!("OUT_DIR"), "/metrics_table.rs"));

const MOVE_FUNCTIONS: [fn(u64) -> u64; 4] = [move_up, move_down, move_right, move_left];

pub fn get_opponent_moves(board: u64) -> impl Iterator<Item = (u64, f64)> {
    let (slot_count, slots) = get_empty_slots(board);

    (0..slot_count).flat_map(move |slot_idx| {
        [(1, 0.9), (2, 0.1)]
            .into_iter()
            .map(move |(cell, probability)| {
                let slot = (slots >> (slot_idx * 4)) & 0xf;
                let new_board = board | (cell << (slot * 4));

                (new_board, probability)
            })
    })
}

const fn mark_empty_cells(board: u64) -> u64 {
    let table = board | (board >> 1);
    let table = table | (table >> 2);

    !table & 0x1111_1111_1111_1111
}

// const fn count_empty_cells(board: u64) -> u32 {
//     let empty_cells = mark_empty_cells(board);

//     empty_cells.count_ones()
// }

fn get_empty_slots(board: u64) -> (u32, u64) {
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

pub fn eval_score(board: u64) -> u32 {
    (0..8).fold(0, |score, i| {
        let cell = ((board >> (i * 8)) & 0xff) as usize;

        score + SCORE_TABLE[cell]
    })
}

// pub fn eval_metrics(board: u64) -> i32 {
//     let row_metrics: i32 = (0..4)
//         .map(|i| -> i32 {
//             let row = (board >> (i * 16)) & 0xffff;

//             METRICS_TABLE[row as usize].into()
//         })
//         .sum();

//     let board = transpose_board(board);

//     let column_metrics: i32 = (0..4)
//         .map(|i| -> i32 {
//             let column = (board >> (i * 16)) & 0xffff;

//             METRICS_TABLE[column as usize].into()
//         })
//         .sum();

//     row_metrics + column_metrics
// }

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
    let board = super::transpose_board(board);

    let new_board = do_move(board);

    super::transpose_board(new_board)
}

fn move_down(board: u64) -> u64 {
    let board = super::transpose_rotate_board(board);

    let new_board = do_move(board);

    super::transpose_rotate_board(new_board)
}

fn move_right(board: u64) -> u64 {
    let board = super::mirror_board(board);

    let new_board = do_move(board);

    super::mirror_board(new_board)
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
