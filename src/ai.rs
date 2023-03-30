use std::{iter::FusedIterator, mem::MaybeUninit, num::NonZeroU64, ops::ControlFlow};

use rand::Rng;

use crate::{direction::Direction, logic};

static mut SCORES: &[u32] = &[];

struct OpponentMoves {
    board: u64,
    slots: u64,
    slot_count: u32,
    current_slot: u32,
}

impl OpponentMoves {
    fn new(board: u64) -> Self {
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
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        (self.current_slot < self.slot_count).then(|| {
            let i = (self.slots >> (self.current_slot * 4)) & 0xf;

            let new_board = self.board | (1 << (i * 4));

            self.current_slot += 1;

            new_board
        })
    }
}

impl ExactSizeIterator for OpponentMoves {
    fn len(&self) -> usize {
        (self.slot_count - self.current_slot) as usize
    }
}

impl FusedIterator for OpponentMoves {}

pub struct Ai<R> {
    rng: R,
    depth: u32,
    iterations: u32,
    move_table: Vec<u16>,
}

impl<R> Ai<R>
where
    R: Rng,
{
    pub fn new(rng: R, depth: u32, iterations: u32) -> Self {
        let scores = (0..u8::MAX)
            .map(|cell_pair| {
                (0..2).fold(0, |score, i| {
                    let exponent = u32::from(cell_pair >> (i * 4)) & 0xf;

                    score
                        + if exponent != 0 {
                            (exponent - 1) * (1 << exponent)
                        } else {
                            0
                        }
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        unsafe {
            SCORES = Box::leak(scores);
        }

        let move_table = (0..u16::MAX).map(Self::move_row).collect();

        let ai = Self {
            rng,
            depth,
            iterations,
            move_table,
        };

        let rows = (0..u64::from(u16::MAX)).filter(|row| {
            !(0..4)
                .map(|i| (row >> (i * 4)) & 0xf)
                .any(|cell| cell == 0xf)
        });

        for row in rows {
            let board = (0..4).fold(0, |board, i| {
                let cell = (row >> (i * 4)) & 0xf;

                board | (cell << (i * 16))
            });

            for i in 0..4 {
                let board = board << (i * 4);

                if ai.try_move(board, Direction::Down)
                    != logic::try_move_down(board).map(|(board, _)| board)
                {
                    panic!("{board}")
                }
            }
        }

        ai
    }

    fn minimax_opponent_move(
        &mut self,
        board: u64,
        depth: u32,
        prune_min: u32,
        prune_max: u32,
    ) -> u32 {
        let mut moves = OpponentMoves::new(board);

        let first_score = moves.next().map_or(0, |board| {
            self.minimax_player_move(board, depth, prune_min, prune_max)
                .map_or(0, |(new_score, _)| new_score)
        });

        if first_score <= prune_min || first_score == 0 {
            first_score
        } else {
            let best = moves.try_fold(first_score, |min_score, board| {
                let new_score = self
                    .minimax_player_move(board, depth, prune_min, min_score)
                    .map_or(0, |(score, _)| score);

                if new_score <= prune_min {
                    ControlFlow::Break(new_score)
                } else if new_score < min_score {
                    ControlFlow::Continue(new_score)
                } else {
                    ControlFlow::Continue(min_score)
                }
            });

            match best {
                ControlFlow::Break(best) | ControlFlow::Continue(best) => best,
            }
        }

        // moves
        //     .map(|board| {
        //         best_player_move(board, depth).map_or((board, 0), |(board, score, _)| (board, score))
        //     })
        //     .min_by_key(|&(_, score)| score)
        //     .unwrap_or((board, 0))
    }

    fn minimax_player_move(
        &mut self,
        board: u64,
        depth: u32,
        prune_min: u32,
        prune_max: u32,
    ) -> Option<(u32, Direction)> {
        let mut moves_array = MaybeUninit::uninit_array::<4>();
        let count = self.get_all_moves(&mut moves_array, board);

        let mut player_moves =
            unsafe { MaybeUninit::slice_assume_init_mut(&mut moves_array[0..count]) }.iter();

        if let Some(depth) = depth.checked_sub(1) {
            let maybe_first_move_result = player_moves.next().map(|&(board, direction)| {
                let new_score = self.minimax_opponent_move(board, depth, prune_min, prune_max);
                (new_score, direction)
            });

            maybe_first_move_result.map(|(first_score, first_direction)| {
                if first_score >= prune_max {
                    (first_score, first_direction)
                } else {
                    let best = player_moves.try_fold(
                        (first_score, first_direction),
                        |(max_score, best_direction), &(board, direction)| {
                            let new_score =
                                self.minimax_opponent_move(board, depth, max_score, prune_max);

                            if new_score >= prune_max {
                                ControlFlow::Break((new_score, direction))
                            } else if new_score > max_score {
                                ControlFlow::Continue((new_score, direction))
                            } else {
                                ControlFlow::Continue((max_score, best_direction))
                            }
                        },
                    );

                    match best {
                        ControlFlow::Break(best) | ControlFlow::Continue(best) => best,
                    }
                }
            })
        } else {
            player_moves
                .map(|&(board, direction)| (Self::eval_score(board), direction))
                .max_by_key(|&(score, _)| score)
        }

        // if let Some(depth) = depth.checked_sub(1) {
        //     player_moves
        //         .map(|(board, score, direction)| {
        //             let (new_board, additional_score) = best_opponent_move(board, depth);
        //             (new_board, score + additional_score, direction)
        //         })
        //         .max_by_key(|&(_, score, _)| score)
        // } else {
        //     player_moves.max_by_key(|&(_, score, _)| score)
        // }
    }

    pub fn get_next_move(&mut self, board: u64) -> Option<Direction> {
        self.minimax_player_move(board, self.depth, 0, u32::MAX)
            .map(|(_, direction)| direction)
    }

    pub fn eval_score(board: u64) -> u32 {
        // const SCORES2: [u32; 16] = [
        //     0, 0, 4, 16, 48, 128, 320, 768, 1792, 4096, 9216, 20480, 45056, 98304, 212992, 458752,
        // ];

        let scores1 = (0..8).fold(0, |score, i| {
            let cell = ((board >> (i * 8)) & 0xff) as usize;

            score + unsafe { SCORES.get_unchecked(cell) }
        });

        // let scores2 = (0..16).fold(0, |score, i| {
        //     let cell = ((board >> (i * 4)) & 0xf) as usize;

        //     score + SCORES2[cell]
        // });

        // let scores3 = (0..16).fold(0, |score, i| {
        //     let exponent = ((board >> (i * 4)) & 0xf) as u32;

        //     score
        //         + if exponent != 0 {
        //             (exponent - 1) * (1 << exponent)
        //         } else {
        //             0
        //         }
        // });

        // if scores1 != scores2 || scores1 != scores3 {
        //     panic!("{board}, {scores1}, {scores2}, {scores3}")
        // }

        scores1
    }

    pub fn eval_monte_carlo(&mut self, board: u64) -> u32 {
        let score_sum: f64 = (0..(self.iterations))
            .map(|_| {
                let final_board = (0..).try_fold(board, |board, _| {
                    let board = logic::spawn_square(&mut self.rng, board);

                    let new_boards_iter =
                        Direction::iter().filter_map(|direction| self.try_move(board, direction));

                    let mut new_boards = MaybeUninit::uninit_array::<4>();
                    let mut count = 0;

                    for new_board in new_boards_iter {
                        new_boards[count] = MaybeUninit::new(new_board.get());
                        count += 1;
                    }

                    let new_boards =
                        unsafe { MaybeUninit::slice_assume_init_ref(&new_boards[0..count]) };

                    if count > 0 {
                        let i = self.rng.gen_range(0..count);

                        ControlFlow::Continue(new_boards[i])
                    } else {
                        ControlFlow::Break(board)
                    }
                });

                let final_board = match final_board {
                    ControlFlow::Break(board) | ControlFlow::Continue(board) => board,
                };

                Self::eval_score(final_board) as f64
            })
            .sum();

        (score_sum / self.iterations as f64) as u32
    }

    fn move_row(mut row: u16) -> u16 {
        let shift = row.trailing_zeros() & !0x3;

        row = row.wrapping_shr(shift);

        let mut mask = 0;

        for j in [0, 4] {
            mask = (mask << 4) | 0xf;

            let mut sub_row = row & !mask;
            row &= mask;
            let shift = sub_row.trailing_zeros() & !0x3;
            sub_row = sub_row.wrapping_shr(shift);

            if sub_row & 0xf == (row >> j) & 0xf && sub_row & 0xf != 0 {
                // This can overflow into if the adjacent square if the square being incremented
                // has reached 15.
                row += 1 << j;
            } else {
                sub_row <<= 4;
            }

            row |= (sub_row << j) & !mask;
        }

        if row >> 12 == (row >> 8) & 0xf && row >> 12 != 0 {
            // This can overflow into if the adjacent square if the square being incremented
            // has reached 15.
            row &= 0xfff;
            row += 1 << 8;
        }

        row
    }

    const fn reverse_rows(board: u64) -> u64 {
        let board = ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0xf0f_0f0f_0f0f_0f0f);
        ((board << 8) & 0xff00_ff00_ff00_ff00) | ((board >> 8) & 0xff_00ff_00ff_00ff)
    }

    fn move_up(&self, board: u64) -> u64 {
        (0..4)
            .map(|i| {
                let column = (0..4).fold(0, |column, j| {
                    let cell = (board >> (j * 12 + i * 4)) as u16 & (0xf << (j * 4));

                    column | cell
                });

                (i, unsafe {
                    *self.move_table.get_unchecked(column as usize)
                })
            })
            .fold(0, |board, (i, column)| {
                let board_column = (0..4).fold(0, |board_column, j| {
                    let cell = (column as u64 & (0xf << (j * 4))) << (j * 12 + i * 4);

                    board_column | cell
                });

                board | board_column
            })
    }

    fn move_down(&self, board: u64) -> u64 {
        (0..4)
            .map(|i| {
                let column = (1..4).fold((board << (12 - i * 4)) as u16 & 0xf000, |column, j| {
                    let cell = (board >> (j * 20 - 12 + i * 4)) as u16 & (0xf000 >> (j * 4));

                    column | cell
                });

                (i, unsafe {
                    *self.move_table.get_unchecked(column as usize)
                })
            })
            .fold(0, |board, (i, column)| {
                let board_column = (1..4).fold(
                    (u64::from(column) & 0xf000) >> (12 - i * 4),
                    |board_column, j| {
                        let cell =
                            (u64::from(column) & (0xf000 >> (j * 4))) << (j * 20 - 12 + i * 4);

                        board_column | cell
                    },
                );

                board | board_column
            })
    }

    fn move_right(&self, board: u64) -> u64 {
        let board = Self::reverse_rows(board);

        let board = self.move_left(board);

        Self::reverse_rows(board)
    }

    fn move_left(&self, board: u64) -> u64 {
        (0..4)
            .map(|i| {
                let row = (board >> (i * 16)) as u16;
                (i, unsafe { *self.move_table.get_unchecked(row as usize) })
            })
            .fold(0, |board, (i, row)| board | (u64::from(row) << (i * 16)))
    }

    fn try_move(&self, board: u64, direction: Direction) -> Option<NonZeroU64> {
        let new_board = match direction {
            Direction::Up => self.move_up(board),
            Direction::Down => self.move_down(board),
            Direction::Right => self.move_right(board),
            Direction::Left => self.move_left(board),
        };

        (new_board != board)
            .then_some(NonZeroU64::new(new_board))
            .flatten()
    }

    fn get_all_moves(
        &mut self,
        moves_out: &mut [MaybeUninit<(u64, Direction)>; 4],
        board: u64,
    ) -> usize {
        let moves_iter = Direction::iter().filter_map(|direction| {
            self.try_move(board, direction)
                .map(|new_board| (new_board.get(), direction))
        });

        let mut count: usize = 0;

        for move_out in moves_iter {
            moves_out[count] = MaybeUninit::new(move_out);
            count += 1;
        }

        count
    }
}

pub fn get_next_move_random(rng: &mut impl Rng, board: u64) -> Option<Direction> {
    let moves: Vec<_> = Direction::iter()
        .filter_map(|direction| logic::try_move(board, direction).map(|_| direction))
        .collect();

    (!moves.is_empty()).then(|| moves[rng.gen_range(0..(moves.len()))])
}
