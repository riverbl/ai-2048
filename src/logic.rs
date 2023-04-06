use std::{
    iter::{FusedIterator, TrustedLen},
    mem::MaybeUninit,
    num::NonZeroU64,
};

use rand::Rng;

use crate::direction::Direction;

static MOVE_TABLE: [u16; 1 << 16] = include!(concat!(env!("OUT_DIR"), "/move_table.rs"));
static SCORE_TABLE: [u32; 1 << 8] = include!(concat!(env!("OUT_DIR"), "/score_table.rs"));
static METRICS_TABLE: [i8; 1 << 16] = include!(concat!(env!("OUT_DIR"), "/metrics_table.rs"));

const MOVE_FUNCTIONS: [fn(u64) -> u64; 4] = [move_up, move_down, move_right, move_left];

pub struct OpponentMoves {
    board: u64,
    slots: u64,
    slot_count: u32,
    current_slot: u32,
}

impl OpponentMoves {
    pub fn new(board: u64) -> Self {
        let mut slots: u64 = 0;
        let mut slot_count = 0;

        for i in 0..16 {
            if (board >> (i * 4)) & 0xf == 0 {
                slots |= i << (slot_count * 4);
                slot_count += 1;
            }
        }

        Self {
            board,
            slots,
            slot_count,
            current_slot: 0,
        }
    }
}

impl Iterator for OpponentMoves {
    type Item = (u64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        (self.current_slot / 2 < self.slot_count).then(|| {
            let i = (self.slots >> ((self.current_slot & !0x1) * 2)) & 0xf;

            let cell = u64::from(self.current_slot & 0x1) + 1;
            let new_board = self.board | (cell << (i * 4));

            let probability = if self.current_slot & 0x1 == 0 {
                0.9
            } else {
                0.1
            };

            self.current_slot += 1;

            (new_board, probability)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }
}

impl ExactSizeIterator for OpponentMoves {
    fn len(&self) -> usize {
        (self.slot_count * 2 - self.current_slot) as usize
    }
}

unsafe impl TrustedLen for OpponentMoves {}

impl FusedIterator for OpponentMoves {}

pub fn spawn_square(rng: &mut impl Rng, board: u64) -> u64 {
    let mut slots: u64 = 0;
    let mut slot_count = 0;

    for i in 0..16 {
        if (board >> (i * 4)) & 0xf == 0 {
            slots |= i << (slot_count * 4);
            slot_count += 1;
        }
    }

    if slot_count > 0 {
        let rand = rng.gen_range(0..(slot_count * 10));

        let slot = rand / 10;
        let cell = if rand % 10 == 0 { 2 } else { 1 };

        let i = (slots >> (slot * 4)) & 0xf;

        board | (cell << (i * 4))
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

// pub fn count_empty_cells(board: u64) -> u32 {
//     let table = board | (board >> 1);
//     let table = table | (table >> 2);

//     let table = !table & 0x1111_1111_1111_1111;

//     let sum = table + (table >> 4);
//     let sum = sum + (sum >> 8);
//     let sum = sum + (sum >> 16);
//     let sum = sum + (sum >> 32);

//     (sum & 0xf) as u32
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
        .map(|try_move| try_move(board))
        .map(|new_board| {
            (new_board != board)
                .then_some(NonZeroU64::new(new_board))
                .flatten()
        })
}

pub fn get_all_moves(moves_out: &mut [MaybeUninit<(u64, Direction)>; 4], board: u64) -> usize {
    let moves_iter = Direction::iter().filter_map(|direction| {
        try_move(board, direction).map(|new_board| (new_board.get(), direction))
    });

    let mut count: usize = 0;

    for move_out in moves_iter {
        moves_out[count] = MaybeUninit::new(move_out);
        count += 1;
    }

    count
}

pub fn try_move(board: u64, direction: Direction) -> Option<NonZeroU64> {
    let new_board = MOVE_FUNCTIONS[direction as usize](board);

    (new_board != board)
        .then_some(NonZeroU64::new(new_board))
        .flatten()
}
