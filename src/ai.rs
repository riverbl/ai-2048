use std::{collections::HashMap, mem::MaybeUninit, ops::ControlFlow};

use rand::Rng;

use crate::{
    control_flow_helper::ControlFlowHelper,
    direction::Direction,
    logic::{self, OpponentMoves},
};

pub struct Ai<R> {
    rng: R,
    depth: u32,
    iterations: u32,
    transposition_table: HashMap<u64, f64>,
}

impl<R> Ai<R>
where
    R: Rng,
{
    pub fn new(rng: R, depth: u32, iterations: u32) -> Self {
        Self {
            rng,
            depth,
            iterations,
            transposition_table: HashMap::new(),
        }
    }

    fn expectimax_opponent_move(&mut self, board: u64, depth: u32) -> f64 {
        let moves = OpponentMoves::new(board);

        let (count, total_score) = moves.fold((0, 0.0), |(count, total_score), board| {
            let maybe_score = self.transposition_table.get(&board).copied();

            let score = maybe_score.unwrap_or_else(|| {
                let score = self
                    .expectimax_player_move(board, depth)
                    .map_or(0.0, |(score, _)| score);

                self.transposition_table.insert(board, score);

                score
            });

            (count + 1, total_score + score)
        });

        total_score / (count as f64)
    }

    fn expectimax_player_move(&mut self, board: u64, depth: u32) -> Option<(f64, Direction)> {
        let mut moves_array = MaybeUninit::uninit_array::<4>();
        let count = logic::get_all_moves(&mut moves_array, board);

        let player_moves =
            unsafe { MaybeUninit::slice_assume_init_mut(&mut moves_array[0..count]) }.iter();

        if let Some(depth) = depth.checked_sub(1) {
            player_moves
                .map(|&(board, direction)| {
                    let maybe_score = self.transposition_table.get(&board).copied();

                    let score = maybe_score.unwrap_or_else(|| {
                        let score = self.expectimax_opponent_move(board, depth);

                        self.transposition_table.insert(board, score);

                        score
                    });

                    (score, direction)
                })
                .max_by(|&(score1, _), &(score2, _)| score1.total_cmp(&score2))
        } else {
            player_moves
                .map(|&(board, direction)| (logic::eval_score(board) as f64, direction))
                .max_by(|&(score1, _), &(score2, _)| score1.total_cmp(&score2))
        }
    }

    fn minimax_opponent_move(
        rng: &mut R,
        iterations: u32,
        board: u64,
        depth: u32,
        prune_min: u32,
        prune_max: u32,
    ) -> u32 {
        let mut moves = OpponentMoves::new(board);

        let first_score = moves.next().map_or(0, |board| {
            Self::minimax_player_move(rng, iterations, board, depth, prune_min, prune_max)
                .map_or(0, |(new_score, _)| new_score)
        });

        if first_score <= prune_min || first_score == 0 {
            first_score
        } else {
            moves
                .try_fold(first_score, |min_score, board| {
                    let new_score = Self::minimax_player_move(
                        rng, iterations, board, depth, prune_min, min_score,
                    )
                    .map_or(0, |(score, _)| score);

                    if new_score <= prune_min {
                        ControlFlow::Break(new_score)
                    } else if new_score < min_score {
                        ControlFlow::Continue(new_score)
                    } else {
                        ControlFlow::Continue(min_score)
                    }
                })
                .into_inner()
        }
    }

    fn minimax_player_move(
        rng: &mut R,
        iterations: u32,
        board: u64,
        depth: u32,
        prune_min: u32,
        prune_max: u32,
    ) -> Option<(u32, Direction)> {
        let mut moves_array = MaybeUninit::uninit_array::<4>();
        let count = logic::get_all_moves(&mut moves_array, board);

        let mut player_moves =
            unsafe { MaybeUninit::slice_assume_init_mut(&mut moves_array[0..count]) }.iter();

        if let Some(depth) = depth.checked_sub(1) {
            let maybe_first_move_result = player_moves.next().map(|&(board, direction)| {
                let new_score = Self::minimax_opponent_move(
                    rng, iterations, board, depth, prune_min, prune_max,
                );
                (new_score, direction)
            });

            maybe_first_move_result.map(|(first_score, first_direction)| {
                if first_score >= prune_max {
                    (first_score, first_direction)
                } else {
                    player_moves
                        .try_fold(
                            (first_score, first_direction),
                            |(max_score, best_direction), &(board, direction)| {
                                let new_score = Self::minimax_opponent_move(
                                    rng, iterations, board, depth, max_score, prune_max,
                                );

                                if new_score >= prune_max {
                                    ControlFlow::Break((new_score, direction))
                                } else if new_score > max_score {
                                    ControlFlow::Continue((new_score, direction))
                                } else {
                                    ControlFlow::Continue((max_score, best_direction))
                                }
                            },
                        )
                        .into_inner()
                }
            })
        } else {
            player_moves
                .map(|&(board, direction)| (logic::eval_score(board), direction))
                .max_by_key(|&(score, _)| score)
        }
    }

    pub fn get_next_move_minimax(&mut self, board: u64) -> Option<Direction> {
        Self::minimax_player_move(
            &mut self.rng,
            self.iterations,
            board,
            self.depth,
            0,
            u32::MAX,
        )
        .map(|(_, direction)| direction)
    }

    pub fn get_next_move_expectimax(&mut self, board: u64) -> Option<Direction> {
        self.transposition_table.clear();

        self.expectimax_player_move(board, self.depth)
            .map(|(_, direction)| direction)
    }

    pub fn get_next_move_monte_carlo(&mut self, board: u64) -> Option<Direction> {
        let mut moves_array = MaybeUninit::uninit_array::<4>();
        let count = logic::get_all_moves(&mut moves_array, board);

        let player_moves =
            unsafe { MaybeUninit::slice_assume_init_ref(&mut moves_array[0..count]) }.iter();

        player_moves
            .max_by_key(|&&(board, _)| {
                Self::eval_monte_carlo(&mut self.rng, self.iterations, board)
            })
            .map(|(_, direction)| direction)
            .copied()
    }

    fn eval_monte_carlo(rng: &mut R, iterations: u32, board: u64) -> u32 {
        let score_sum: f64 = (0..iterations)
            .map(|_| {
                let final_board = (0..)
                    .try_fold(board, |board, _| {
                        let board = logic::spawn_square(rng, board);

                        let new_boards_iter = Direction::iter()
                            .filter_map(|direction| logic::try_move(board, direction));

                        let mut new_boards = MaybeUninit::uninit_array::<4>();
                        let mut count = 0;

                        for new_board in new_boards_iter {
                            new_boards[count] = MaybeUninit::new(new_board.get());
                            count += 1;
                        }

                        let new_boards =
                            unsafe { MaybeUninit::slice_assume_init_ref(&new_boards[0..count]) };

                        if count > 0 {
                            let i = rng.gen_range(0..count);

                            ControlFlow::Continue(new_boards[i])
                        } else {
                            ControlFlow::Break(board)
                        }
                    })
                    .into_inner();

                logic::eval_score(final_board) as f64
            })
            .sum();

        (score_sum / iterations as f64) as u32
    }

    fn eval_monte_carlo2(rng: &mut R, iterations: u32, board: u64) -> u32 {
        let score_sum: f64 = (0..iterations)
            .map(|_| {
                let final_board = (0..)
                    .try_fold(board, |board, _| {
                        let board = logic::spawn_square(rng, board);

                        let new_boards_iter = Direction::iter()
                            .filter_map(|direction| logic::try_move(board, direction));

                        let mut new_boards = MaybeUninit::uninit_array::<4>();
                        let mut count = 0;

                        for new_board in new_boards_iter {
                            new_boards[count] = MaybeUninit::new(new_board.get());
                            count += 1;
                        }

                        let new_boards =
                            unsafe { MaybeUninit::slice_assume_init_ref(&new_boards[0..count]) }
                                .iter()
                                .copied();

                        let maybe_best_board = new_boards
                            .max_by_key(|&new_board| Self::eval_monte_carlo(rng, 3, new_board));

                        maybe_best_board.map_or(ControlFlow::Break(board), |best_board| {
                            ControlFlow::Continue(best_board)
                        })
                    })
                    .into_inner();

                logic::eval_score(final_board) as f64
            })
            .sum();

        (score_sum / iterations as f64) as u32
    }
}
