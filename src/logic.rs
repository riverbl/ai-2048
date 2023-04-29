use std::num::NonZeroU64;

use rand::Rng;

use crate::direction::Direction;

static MOVE_TABLE: [u16; 1 << 16] = include!(concat!(env!("OUT_DIR"), "/move_table.rs"));
static SCORE_TABLE: [u32; 1 << 8] = include!(concat!(env!("OUT_DIR"), "/score_table.rs"));
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

const fn count_empty_cells(board: u64) -> u32 {
    let empty_cells = mark_empty_cells(board);

    empty_cells.count_ones()
}

const fn get_empty_slots(board: u64) -> (u32, u64) {
    let mut slots: u64 = 0;
    let mut slot_count = 0;

    let mut empty_cells = mark_empty_cells(board);

    while empty_cells != 0 {
        let trailing_zeros = empty_cells.trailing_zeros();

        let slot = trailing_zeros as u64 / 4;
        empty_cells &= empty_cells - 1;

        slots |= slot << slot_count;
        slot_count += 4;
    }

    slot_count /= 4;

    (slot_count, slots)
}

pub fn spawn_square(rng: &mut impl Rng, board: u64) -> u64 {
    let slot_count = count_empty_cells(board);

    if slot_count > 0 {
        let rand = rng.gen_range(0..(slot_count * 10));

        let slot_idx = rand / 10;
        let cell = if rand % 10 == 0 { 2 } else { 1 };

        let empty_cells = mark_empty_cells(board);

        let init_slot = empty_cells.trailing_zeros();

        let slot = (0..slot_idx).fold(init_slot, |slot, _| {
            let slot = slot + 4;

            slot + (empty_cells >> slot).trailing_zeros()
        });

        board | (cell << slot)
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

pub const fn mirror_board(board: u64) -> u64 {
    let board = ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    ((board << 8) & 0xff00_ff00_ff00_ff00) | ((board >> 8) & 0x00ff_00ff_00ff_00ff)
}

pub const fn transpose_board(board: u64) -> u64 {
    let keep = board & 0xf0f0_0f0f_f0f0_0f0f;
    let left = board & 0x0000_f0f0_0000_f0f0;
    let right = board & 0x0f0f_0000_0f0f_0000;
    let board = keep | (left << 12) | (right >> 12);

    let keep = board & 0xff00_ff00_00ff_00ff;
    let left = board & 0x0000_0000_ff00_ff00;
    let right = board & 0x00ff_00ff_0000_0000;

    keep | (left << 24) | (right >> 24)
}

pub const fn transpose_rotate_board(board: u64) -> u64 {
    let keep = board & 0x0f0f_f0f0_0f0f_f0f0;
    let left = board & 0x0000_0f0f_0000_0f0f;
    let right = board & 0xf0f0_0000_f0f0_0000;
    let board = keep | (left << 20) | (right >> 20);

    let keep = board & 0x00ff_00ff_ff00_ff00;
    let left = board & 0x0000_0000_00ff_00ff;
    let right = board & 0xff00_ff00_0000_0000;

    keep | (left << 40) | (right >> 40)
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
    let board = transpose_board(board);

    let new_board = do_move(board);

    transpose_board(new_board)
}

fn move_down(board: u64) -> u64 {
    let board = transpose_rotate_board(board);

    let new_board = do_move(board);

    transpose_rotate_board(new_board)
}

fn move_right(board: u64) -> u64 {
    let board = mirror_board(board);

    let new_board = do_move(board);

    mirror_board(new_board)
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
