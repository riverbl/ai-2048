use std::{collections::HashMap, mem::MaybeUninit};

use crate::{
    direction::Direction,
    logic::{self, OpponentMoves},
};

use super::Ai;

pub struct ExpectimaxAi {
    depth: u32,
    transposition_table: HashMap<(u64, u32), f64>,
}

impl Ai for ExpectimaxAi {
    fn get_next_move(&mut self, board: u64) -> Option<Direction> {
        self.transposition_table.clear();

        self.expectimax_player_move(board, self.depth)
            .map(|(_, direction)| direction)
    }
}

impl ExpectimaxAi {
    pub fn new(depth: u32) -> Self {
        Self {
            depth,
            transposition_table: HashMap::new(),
        }
    }

    fn expectimax_opponent_move(&mut self, board: u64, depth: u32) -> f64 {
        let moves = OpponentMoves::new(board);

        let (total_probability, total_score) = moves.fold(
            (0.0, 0.0),
            |(total_probability, total_score), (board, probability)| {
                let maybe_score = self
                    .transposition_table
                    .get(&(board, depth * 2 + 1))
                    .copied();
                // let maybe_score = None;

                let weighted_score = maybe_score.unwrap_or_else(|| {
                    let score = self
                        .expectimax_player_move(board, depth)
                        .map_or_else(|| f64::from(logic::eval_score(board)), |(score, _)| score);

                    self.transposition_table
                        .insert((board, depth * 2 + 1), score);

                    score
                }) * probability;

                (
                    total_probability + probability,
                    total_score + weighted_score,
                )
            },
        );

        total_score / total_probability
    }

    fn expectimax_player_move(&mut self, board: u64, depth: u32) -> Option<(f64, Direction)> {
        let mut moves_array = MaybeUninit::uninit_array::<4>();
        let count = logic::get_all_moves(&mut moves_array, board);

        let player_moves =
            unsafe { MaybeUninit::slice_assume_init_mut(&mut moves_array[0..count]) }.iter();

        if let Some(depth) = depth.checked_sub(1) {
            player_moves
                .map(|&(board, direction)| {
                    // let maybe_score = self.transposition_table.get(&(board, depth * 2)).copied();
                    let maybe_score = None;

                    let score = maybe_score.unwrap_or_else(|| {
                        let score = self.expectimax_opponent_move(board, depth);

                        // self.transposition_table.insert((board, depth * 2), score);

                        score
                    });

                    (score, direction)
                })
                .max_by(|&(score1, _), &(score2, _)| score1.total_cmp(&score2))
        } else {
            player_moves
                .map(|&(board, direction)| (f64::from(logic::eval_score(board)), direction))
                .max_by(|&(score1, _), &(score2, _)| score1.total_cmp(&score2))
        }
    }
}
