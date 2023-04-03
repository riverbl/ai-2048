use std::{
    iter::{FusedIterator, TrustedLen},
    mem::MaybeUninit,
    num::NonZeroU64,
};

use rand::Rng;

use crate::direction::Direction;

static MOVE_TABLE: [u16; 1 << 16] = include!(concat!(env!("OUT_DIR"), "/move_table.rs"));
static SCORE_TABLE: [u32; 1 << 8] = include!(concat!(env!("OUT_DIR"), "/score_table.rs"));
static EMPTY_CELL_COUNT_TABLE: [u8; 1 << 8] =
    include!(concat!(env!("OUT_DIR"), "/empty_cell_count_table.rs"));

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
                1.0
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
    // const SCORES2: [u32; 16] = [
    //     0, 0, 4, 16, 48, 128, 320, 768, 1792, 4096, 9216, 20480, 45056, 98304, 212992, 458752,
    // ];

    (0..8).fold(0, |score, i| {
        let cell = ((board >> (i * 8)) & 0xff) as usize;

        score + SCORE_TABLE[cell]
    })

    // (0..16).fold(0, |score, i| {
    //     let exponent = ((board >> (i * 4)) & 0xf) as u32;

    //     score
    //         + if exponent != 0 {
    //             (exponent - 1) * (1 << exponent)
    //         } else {
    //             0
    //         }
    // })

    // if scores1 != scores2 || scores1 != scores3 {
    //     panic!("{board}, {scores1}, {scores2}, {scores3}")
    // }
}

fn count_empty_cells(board: u64) -> u32 {
    (0..8).fold(0, |empty_cell_count, i| {
        let cell = ((board >> (i * 8)) & 0xff) as usize;

        empty_cell_count + u32::from(EMPTY_CELL_COUNT_TABLE[cell])
    })
}

fn move_up(board: u64) -> u64 {
    (0..4)
        .map(|i| {
            let column = (0..4).fold(0, |column, j| {
                let cell = (board >> (j * 12 + i * 4)) as u16 & (0xf << (j * 4));

                column | cell
            });

            (i, MOVE_TABLE[column as usize])
        })
        .fold(0, |new_board, (i, column)| {
            let board_column = (0..4).fold(0, |board_column, j| {
                let cell = (column as u64 & (0xf << (j * 4))) << (j * 12 + i * 4);

                board_column | cell
            });

            new_board | board_column
        })
}

fn move_down(board: u64) -> u64 {
    (0..4)
        .map(|i| {
            let column = (1..4).fold((board << (12 - i * 4)) as u16 & 0xf000, |column, j| {
                let cell = (board >> (j * 20 - 12 + i * 4)) as u16 & (0xf000 >> (j * 4));

                column | cell
            });

            (i, MOVE_TABLE[column as usize])
        })
        .fold(0, |new_board, (i, column)| {
            let board_column = (1..4).fold(
                (u64::from(column) & 0xf000) >> (12 - i * 4),
                |board_column, j| {
                    let cell = (u64::from(column) & (0xf000 >> (j * 4))) << (j * 20 - 12 + i * 4);

                    board_column | cell
                },
            );

            new_board | board_column
        })
}

const fn reverse_rows(board: u64) -> u64 {
    let board = ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0xf0f_0f0f_0f0f_0f0f);
    ((board << 8) & 0xff00_ff00_ff00_ff00) | ((board >> 8) & 0xff_00ff_00ff_00ff)
}

fn move_right(board: u64) -> u64 {
    let board = reverse_rows(board);

    let new_board = move_left(board);

    reverse_rows(new_board)
}

fn move_left(board: u64) -> u64 {
    (0..4)
        .map(|i| {
            let row = (board >> (i * 16)) as u16;
            (i, MOVE_TABLE[row as usize])
        })
        .fold(0, |new_board, (i, row)| {
            new_board | (u64::from(row) << (i * 16))
        })
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
