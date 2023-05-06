use std::ops::ControlFlow;

pub fn loop_try_fold<F, B, C>(init: C, mut f: F) -> B
where
    F: FnMut(C) -> ControlFlow<B, C>,
{
    let mut accum = init;

    loop {
        match f(accum) {
            ControlFlow::Continue(c) => accum = c,
            ControlFlow::Break(b) => break b,
        }
    }
}
