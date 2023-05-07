#![feature(array_windows)]

pub mod metrics;

/// Returns the result of rotating `board` by 180 degrees.
/// # Examples
/// ```
/// use core_2048::rotate_board;
///
/// assert_eq!(rotate_board(0xfedc_ba98_7654_3210), 0x0123_4567_89ab_cdef);
/// ```
pub const fn rotate_board(board: u64) -> u64 {
    let board = board.swap_bytes();

    ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0x0f0f_0f0f_0f0f_0f0f)
}

/// Returns the result of reflecting `board` in the vertical axis.
/// # Examples
/// ```
/// use core_2048::mirror_board;
///
/// assert_eq!(mirror_board(0xfedc_ba98_7654_3210), 0xcdef_89ab_4567_0123);
/// ```
pub const fn mirror_board(board: u64) -> u64 {
    let board = ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
    ((board << 8) & 0xff00_ff00_ff00_ff00) | ((board >> 8) & 0x00ff_00ff_00ff_00ff)
}

/// Returns the result of reflecting `board` in the diagonal axis that leaves the most and least
/// significant nibble unchanged.
/// # Examples
/// ```
/// use core_2048::transpose_board;
///
/// assert_eq!(transpose_board(0xfedc_ba98_7654_3210), 0xfb73_ea62_d951_c840);
/// ```
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

/// Returns the result of reflecting `board` in the diagonal axis that swaps the most and least
/// significant nibbles.
/// # Examples
/// ```
/// use core_2048::{rotate_board, transpose_board, transpose_rotate_board};
///
/// let board = 0xfedc_ba98_7654_3210;
/// let transpose_rotated = transpose_rotate_board(board);
/// assert_eq!(transpose_rotated, 0x048c_159d_26ae_37bf);
/// assert_eq!(transpose_rotated, transpose_board(rotate_board(board)));
/// assert_eq!(transpose_rotated, rotate_board(transpose_board(board)));
/// ```
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
