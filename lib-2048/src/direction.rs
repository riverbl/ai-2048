#[derive(Clone, Copy, Debug)]
pub enum Direction {
    Up = 0,
    Down = 1,
    Right = 2,
    Left = 3,
}

impl Direction {
    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::Up, Self::Down, Self::Right, Self::Left].into_iter()
    }
}
