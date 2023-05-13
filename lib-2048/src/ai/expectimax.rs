use rustc_hash::FxHashMap;

use crate::{direction::Direction, logic};

use super::Ai;

pub struct ExpectimaxAi<F> {
    probability_cutoff: f64,
    transposition_table: FxHashMap<u64, (f64, f64)>,
    loss_weight: f64,
    eval_function: F,
}

impl<F: FnMut(u64) -> f64> Ai for ExpectimaxAi<F> {
    fn get_next_move(&mut self, board: u64) -> Option<Direction> {
        self.transposition_table.clear();

        self.expectimax_player_move(board, 1.0)
            .map(|(_, direction)| direction)
    }
}

impl<F: FnMut(u64) -> f64> ExpectimaxAi<F> {
    pub fn new(probability_cutoff: f64, loss_weight: f64, eval_function: F) -> Self {
        Self {
            probability_cutoff,
            loss_weight,
            transposition_table: FxHashMap::default(),
            eval_function,
        }
    }

    fn expectimax_opponent_move(&mut self, board: u64, probability: f64) -> f64 {
        let moves = logic::get_opponent_moves(board);

        moves.fold(0.0, |total_score, (board, move_probability)| {
            let position_probability = move_probability * probability;

            let maybe_score = (position_probability > self.probability_cutoff)
                .then(|| {
                    self.transposition_table
                        .get(&board)
                        .filter(|&&(_, stored_probability)| stored_probability >= probability)
                        .map(|&(score, _)| score)
                })
                .flatten();
            // let maybe_score = None;

            let score = maybe_score.unwrap_or_else(|| {
                let score = self
                    .expectimax_player_move(board, position_probability)
                    .map_or(self.loss_weight, |(score, _)| score);

                if position_probability > self.probability_cutoff {
                    self.transposition_table.insert(board, (score, probability));
                }

                score
            });

            score.mul_add(move_probability, total_score)
        })
    }

    fn expectimax_player_move(&mut self, board: u64, probability: f64) -> Option<(f64, Direction)> {
        let player_moves = super::get_all_moves(board);

        if probability > self.probability_cutoff {
            player_moves
                .map(|(board, direction)| {
                    let maybe_score = self
                        .transposition_table
                        .get(&board)
                        .filter(|&&(_, stored_probability)| stored_probability >= probability)
                        .map(|&(score, _)| score);
                    // let maybe_score = None;

                    let score = maybe_score.unwrap_or_else(|| {
                        let score = self.expectimax_opponent_move(board, probability * 0.5);

                        self.transposition_table.insert(board, (score, probability));

                        score
                    });

                    (score, direction)
                })
                .max_by(|(score1, _), (score2, _)| score1.total_cmp(score2))
        } else {
            player_moves
                .map(|(board, direction)| ((self.eval_function)(board), direction))
                .max_by(|(score1, _), (score2, _)| score1.total_cmp(score2))
        }
    }
}
