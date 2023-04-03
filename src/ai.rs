use crate::direction::Direction;

pub mod expectimax;
pub mod monte_carlo;
pub mod random;

pub trait Ai {
    fn get_next_move(&mut self, board: u64) -> Option<Direction>;
}
