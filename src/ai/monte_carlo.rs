use std::{mem::MaybeUninit, ops::ControlFlow};

use rand::Rng;

use crate::{control_flow_helper, direction::Direction, logic};

use super::Ai;

pub struct MonteCarloAi<R> {
    rng: R,
    iterations: u32,
}

impl<R> Ai for MonteCarloAi<R>
where
    R: Rng,
{
    fn get_next_move(&mut self, board: u64) -> Option<Direction> {
        let player_moves = super::get_all_moves(board);

        player_moves
            .max_by_key(|&(board, _)| Self::eval_monte_carlo(&mut self.rng, self.iterations, board))
            .map(|(_, direction)| direction)
    }
}

impl<R> MonteCarloAi<R>
where
    R: Rng,
{
    pub const fn new(rng: R, iterations: u32) -> Self {
        Self { rng, iterations }
    }

    fn eval_monte_carlo(rng: &mut R, iterations: u32, board: u64) -> u64 {
        (0..iterations)
            .map(|_| -> u64 {
                let final_board = control_flow_helper::loop_try_fold(board, |board| {
                    let board = logic::spawn_square(rng, board);

                    let new_boards_iter =
                        Direction::iter().filter_map(|direction| super::try_move(board, direction));

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
                });

                logic::eval_score(final_board).into()
            })
            .sum()
    }

    // fn eval_monte_carlo2(rng: &mut R, iterations: u32, board: u64) -> u64 {
    //     (0..iterations)
    //         .map(|_| -> u64 {
    //             let final_board = control_flow_helper::loop_try_fold(board, |board| {
    //                 let board = logic::spawn_square(rng, board);

    //                 let new_boards_iter =
    //                     Direction::iter().filter_map(|direction| logic::try_move(board, direction));

    //                 let mut new_boards = MaybeUninit::uninit_array::<4>();
    //                 let mut count = 0;

    //                 for new_board in new_boards_iter {
    //                     new_boards[count] = MaybeUninit::new(new_board.get());
    //                     count += 1;
    //                 }

    //                 let new_boards =
    //                     unsafe { MaybeUninit::slice_assume_init_ref(&new_boards[0..count]) }
    //                         .iter()
    //                         .copied();

    //                 let maybe_best_board = new_boards
    //                     .max_by_key(|&new_board| Self::eval_monte_carlo(rng, 3, new_board));

    //                 maybe_best_board.map_or(ControlFlow::Break(board), |best_board| {
    //                     ControlFlow::Continue(best_board)
    //                 })
    //             });

    //             logic::eval_score(final_board).into()
    //         })
    //         .sum()
    // }
}
