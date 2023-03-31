use std::ops::ControlFlow;

pub trait ControlFlowHelper<T> {
    fn into_inner(self) -> T;
}

impl<T> ControlFlowHelper<T> for ControlFlow<T, T> {
    fn into_inner(self) -> T {
        match self {
            Self::Break(inner) | Self::Continue(inner) => inner,
        }
    }
}
