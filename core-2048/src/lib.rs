#![feature(array_windows)]

pub mod metrics;

pub const fn mirror_board(board: u64) -> u64 {
    let board = ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    ((board << 8) & 0xff00_ff00_ff00_ff00) | ((board >> 8) & 0x00ff_00ff_00ff_00ff)
}

pub const fn transpose_board(board: u64) -> u64 {
    let keep = board & 0xf0f0_0f0f_f0f0_0f0f;
    let left = board & 0x0000_f0f0_0000_f0f0;
    let right = board & 0x0f0f_0000_0f0f_0000;
    let board = keep | (left << 12) | (right >> 12);

    let keep = board & 0xff00_ff00_00ff_00ff;
    let left = board & 0x0000_0000_ff00_ff00;
    let right = board & 0x00ff_00ff_0000_0000;

    keep | (left << 24) | (right >> 24)
}

pub const fn transpose_rotate_board(board: u64) -> u64 {
    let keep = board & 0x0f0f_f0f0_0f0f_f0f0;
    let left = board & 0x0000_0f0f_0000_0f0f;
    let right = board & 0xf0f0_0000_f0f0_0000;
    let board = keep | (left << 20) | (right >> 20);

    let keep = board & 0x00ff_00ff_ff00_ff00;
    let left = board & 0x0000_0000_00ff_00ff;
    let right = board & 0xff00_ff00_0000_0000;

    keep | (left << 40) | (right >> 40)
}
