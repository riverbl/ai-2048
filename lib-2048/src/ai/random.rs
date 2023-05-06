use std::mem::MaybeUninit;

use rand::Rng;

use crate::direction::Direction;

use super::Ai;

pub struct RandomAi<R> {
    rng: R,
}

impl<R> Ai for RandomAi<R>
where
    R: Rng,
{
    fn get_next_move(&mut self, board: u64) -> Option<Direction> {
        let mut move_array = MaybeUninit::uninit_array::<4>();

        let moves = Direction::iter()
            .filter_map(|direction| super::try_move(board, direction).map(|_| direction));
        let mut count = 0;

        for direction in moves {
            move_array[count] = MaybeUninit::new(direction);
            count += 1;
        }

        (count > 0).then(|| {
            let moves = unsafe { MaybeUninit::slice_assume_init_ref(&move_array[0..count]) };

            moves[self.rng.gen_range(0..count)]
        })
    }
}

impl<R> RandomAi<R>
where
    R: Rng,
{
    pub const fn new(rng: R) -> Self {
        Self { rng }
    }
}
