use rand::Rng;

pub fn spawn_square(rng: &mut impl Rng, board: u64) -> u64 {
    let mut slots: u64 = 0;
    let mut slot_count = 0;

    for i in 0..16 {
        if (board >> (i * 4)) & 0xf == 0 {
            slots |= i << (slot_count * 4);
            slot_count += 1;
        }
    }

    if slot_count > 0 {
        let slot = rng.gen_range(0..slot_count);

        let i = (slots >> (slot * 4)) & 0xf;

        board | (1 << (i * 4))
    } else {
        board
    }
}

pub fn try_move_left(board: u64) -> Option<(u64, u32)> {
    let mut new_board = 0;
    let mut score = 0;

    for i in 0..4 {
        let mut row = (board >> (i * 16)) as u16;
        let shift = row.trailing_zeros() & !0x3;

        row = row.wrapping_shr(shift);

        let mut mask = 0;

        for j in [0, 4] {
            mask = (mask << 4) | 0xf;

            let mut sub_row = row & !mask;
            let shift = sub_row.trailing_zeros() & !0x3;
            sub_row = sub_row.wrapping_shr(shift);

            if sub_row & 0xf == (row >> j) & 0xf && sub_row & 0xf != 0 {
                // This can overflow into if the adjacent square if the square being incremented
                // has reached 15.
                row += 1 << j;

                let score_exponent = (row >> j) & 0xf;
                score += 1 << score_exponent;
            } else {
                sub_row <<= 4;
            }

            row = ((sub_row << j) & !mask) | (row & mask);
        }

        if row >> 12 == (row >> 8) & 0xf && row >> 12 != 0 {
            // This can overflow into if the adjacent square if the square being incremented
            // has reached 15.
            row += 1 << 8;
            row &= 0xfff;

            let score_exponent = (row >> 8) & 0xf;
            score += 1 << score_exponent;
        }

        new_board |= u64::from(row) << (i * 16);
    }

    (new_board != board).then_some((new_board, score))
}

pub fn try_move_right(board: u64) -> Option<(u64, u32)> {
    let mut new_board = 0;
    let mut score = 0;

    for i in 0..4 {
        let mut row = (board >> (i * 16)) as u16;
        let shift = row.leading_zeros() & !0x3;

        row = row.wrapping_shl(shift);

        let mut mask = 0;

        for j in [0, 4] {
            mask = (mask >> 4) | 0xf000;

            let mut sub_row = row & !mask;
            let shift = sub_row.leading_zeros() & !0x3;
            sub_row = sub_row.wrapping_shl(shift);

            if sub_row & 0xf000 == (row << j) & 0xf000 && sub_row & 0xf000 != 0 {
                // This can overflow into if the adjacent square if the square being incremented
                // has reached 15.
                row += 1 << (12 - j);

                let score_exponent = (row >> (12 - j)) & 0xf;
                score += 1 << score_exponent;
            } else {
                sub_row >>= 4;
            }

            row = ((sub_row >> j) & !mask) | (row & mask);
        }

        if row << 12 == (row << 8) & 0xf000 && row << 12 != 0 {
            // This can overflow into if the adjacent square if the square being incremented
            // has reached 15.
            row += 1 << 4;
            row &= 0xfff0;

            let score_exponent = (row >> 4) & 0xf;
            score += 1 << score_exponent;
        }

        new_board |= u64::from(row) << (i * 16);
    }

    (new_board != board).then_some((new_board, score))
}

pub fn try_move_up(board: u64) -> Option<(u64, u32)> {
    let mut new_board = 0;
    let mut score = 0;

    for i in 0..4 {
        let mut column = (board >> (i * 4)) & 0xf_000f_000f_000f;
        let shift = column.trailing_zeros() & !0xf;

        column = column.wrapping_shr(shift);

        let mut mask = 0;

        for j in [0, 16] {
            mask = (mask << 16) | 0xf;

            let mut sub_column = column & !mask;
            let shift = sub_column.trailing_zeros() & !0xf;
            sub_column = sub_column.wrapping_shr(shift);

            if sub_column & 0xf == (column >> j) & 0xf && sub_column & 0xf != 0 {
                // This can overflow into if the adjacent square if the square being incremented
                // has reached 15.
                column += 1 << j;

                let score_exponent = (column >> j) & 0xf;
                score += 1 << score_exponent;
            } else {
                sub_column <<= 16;
            }

            column = ((sub_column << j) & !mask) | (column & mask);
        }

        if column >> 48 == (column >> 32) & 0xf && column >> 48 != 0 {
            // This can overflow into if the adjacent square if the square being incremented
            // has reached 15.
            column += 1 << 32;
            column &= 0xf_000f_000f;

            let score_exponent = (column >> 32) & 0xf;
            score += 1 << score_exponent;
        }

        new_board |= column << (i * 4);
    }

    (new_board != board).then_some((new_board, score))
}

pub fn try_move_down(board: u64) -> Option<(u64, u32)> {
    let mut new_board = 0;
    let mut score = 0;

    for i in 0..4 {
        const FIRST_NIBBLE_MASK: u64 = 0xf000_0000_0000_0000;
        let mut column = (board << (i * 4)) & 0xf000_f000_f000_f000;
        let shift = column.leading_zeros() & !0xf;

        column = column.wrapping_shl(shift);

        let mut mask = 0;

        for j in [0, 16] {
            mask = (mask >> 16) | FIRST_NIBBLE_MASK;

            let mut sub_column = column & !mask;
            let shift = sub_column.leading_zeros() & !0xf;
            sub_column = sub_column.wrapping_shl(shift);

            if sub_column & FIRST_NIBBLE_MASK == (column << j) & FIRST_NIBBLE_MASK
                && sub_column & FIRST_NIBBLE_MASK != 0
            {
                // This can overflow into if the adjacent square if the square being incremented
                // has reached 15.
                column += 1 << (60 - j);

                let score_exponent = (column >> (60 - j)) & 0xf;
                score += 1 << score_exponent;
            } else {
                sub_column >>= 16;
            }

            column = ((sub_column >> j) & !mask) | (column & mask);
        }

        if column << 48 == (column << 32) & FIRST_NIBBLE_MASK && column << 48 != 0 {
            // This can overflow into if the adjacent square if the square being incremented
            // has reached 15.
            column += 1 << 28;
            column &= 0xf000_f000_f000_0000;

            let score_exponent = (column >> 28) & 0xf;
            score += 1 << score_exponent;
        }

        new_board |= column >> (i * 4);
    }

    (new_board != board).then_some((new_board, score))
}

pub fn try_all_moves(board: u64) -> [Option<(u64, u32)>; 4] {
    [try_move_up, try_move_down, try_move_right, try_move_left].map(|try_move| try_move(board))
}
