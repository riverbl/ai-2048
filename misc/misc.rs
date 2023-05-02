// This file contains functions that were not performant or not practical enough to use, but may be
// useful in the future.
// They may have bugs!

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

pub fn spawn_square(rng: &mut impl Rng, board: u64) -> u64 {
    let empty_cells = mark_empty_cells(board);
    let slot_count = empty_cells.count_ones();

    if slot_count > 0 {
        let rand = rng.gen_range(0..(slot_count * 10));

        let slot_idx = rand / 10;
        let cell = if rand % 10 == 0 { 2 } else { 1 };

        // Set low_bits_cleared to empty_cells with the slot_idx lowest set bits cleared.
        let bits = !0 << slot_idx;
        let low_bits_cleared = unsafe { _pdep_u64(bits, empty_cells) };

        let slot = low_bits_cleared.trailing_zeros();

        board | (cell << slot)
    } else {
        board
    }
}

/// Returns the cells in board packed 1 cell per byte over 128 bits, rather than 1 cell per nibble
/// over 64 bits.
fn unpack_board(board: u64) -> __m128i {
    unsafe {
        let board = _mm_cvtsi64_si128(board as _);

        let high = _mm_srli_epi32::<4>(board);

        // Alternate bytes from board and high.
        let cells_with_garbage = _mm_unpacklo_epi8(board, high);

        _mm_and_si128(cells_with_garbage, _mm_set1_epi8(0xf))
    }
}

pub fn eval_score(board: u64) -> u32 {
    unsafe {
        let cells = unpack_board(board);

        let zeros = _mm256_set1_epi64x(0);

        let cells = {
            let zeros = _mm_set1_epi64x(0);
            let low = _mm_unpacklo_epi8(cells, zeros);
            let high = _mm_unpackhi_epi8(cells, zeros);

            _mm256_set_m128i(high, low)
        };

        let ones = _mm256_set1_epi16(1);
        let all_ones = _mm256_set1_epi32(0xffff_ffffu32 as _);

        let scores = {
            // Needs AVX-512.
            let values = _mm256_sllv_epi16(all_ones, cells);
            let counts = _mm256_subs_epu16(cells, ones);

            _mm256_madd_epi16(values, counts)
        };

        let scores = {
            let low_scores = _mm256_castsi256_si128(scores);
            let high_scores = _mm256_extracti128_si256::<1>(scores);

            _mm_add_epi32(low_scores, high_scores)
        };

        let scores = {
            let high_scores = _mm_unpackhi_epi64(scores, scores);

            _mm_add_epi32(scores, high_scores)
        };

        let score = {
            let high_score = _mm_shuffle_epi32::<0x55>(scores);

            _mm_add_epi32(scores, high_score)
        };

        (-_mm_cvtsi128_si32(score)) as _
    }
}

pub fn eval_score(board: u64) -> u32 {
    unsafe {
        let cells = unpack_board(board);

        let cells = {
            let zeros = _mm_set1_epi64x(0);
            let low = _mm_unpacklo_epi8(cells, zeros);
            let high = _mm_unpackhi_epi8(cells, zeros);

            _mm256_set_m128i(high, low)
        };

        let zeros = _mm256_set1_epi64x(0);
        let low = _mm256_unpacklo_epi16(cells, zeros);
        let high = _mm256_unpackhi_epi16(cells, zeros);

        let ones = _mm256_set1_epi32(1);

        let [low_scores, high_scores] = [low, high].map(|cells| {
            let mask = _mm256_cmpeq_epi32(cells, zeros);

            let values = _mm256_sllv_epi32(ones, cells);
            let counts = {
                let wrapping_sub = _mm256_sub_epi32(cells, ones);
                _mm256_andnot_si256(mask, wrapping_sub)
            };

            _mm256_mullo_epi32(values, counts)
        });

        let scores = _mm256_add_epi32(low_scores, high_scores);

        let scores = {
            let low_scores = _mm256_castsi256_si128(scores);
            let high_scores = _mm256_extracti128_si256::<1>(scores);

            _mm_add_epi32(low_scores, high_scores)
        };

        let scores = {
            let high_scores = _mm_unpackhi_epi64(scores, scores);

            _mm_add_epi32(scores, high_scores)
        };

        let score = {
            let high_score = _mm_shuffle_epi32::<0x55>(scores);

            _mm_add_epi32(scores, high_score)
        };

        _mm_cvtsi128_si32(score) as _
    }
}

pub fn mirror_board(board: u64) -> u64 {
    unsafe {
        let board = ((board << 4) & 0xf0f0_f0f0_f0f0_f0f0) | ((board >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
        let board = _mm_cvtsi64_si128(board as _);
        let mask = _mm_cvtsi64_si128(0x0607_0405_0203_0001);

        let shuffled = _mm_shuffle_epi8(board, mask);

        _mm_cvtsi128_si64(shuffled) as _
    }
}

pub fn vertical_mirror_board(board: u64) -> u64 {
    unsafe {
        let board = _mm_cvtsi64_si128(board as _);

        let shuffled = _mm_shufflelo_epi16::<0x1b>(board);

        _mm_cvtsi128_si64(shuffled) as _
    }
}

pub fn transpose_board(board: u64) -> u64 {
    unsafe {
        // The low 64 bits of cells contains nibbles that are transformed to the same position
        // within the byte (low -> low or high -> high).
        // The high 64 bits contain nibbles that are transformed low -> high or high -> low.
        let cells = _mm_set_epi64x(
            (((board & 0x0000_f0f0_0000_f0f0) >> 4) | ((board & 0x0f0f_0000_0f0f_0000) << 4)) as _,
            (board & 0xf0f0_0f0f_f0f0_0f0f) as _,
        );
        let mask = _mm_set_epi64x(0x0d09_0f0b_0c08_0e0au64 as _, 0x0703_0501_0602_0400u64 as _);

        let shuffled = _mm_shuffle_epi8(cells, mask);

        // The high and low 64 bits of high both contain the high 64 bits of shuffled.
        let high = {
            let shuffled = _mm256_castsi128_si256(shuffled);

            let high = _mm256_shuffle_epi32::<0xee>(shuffled);
            _mm256_castsi256_si128(high)
        };

        let transposed_board = _mm_or_si128(shuffled, high);

        _mm_cvtsi128_si64(transposed_board) as _
    }
}

/// Permutes the nibbles of board according to HIGH and LOW.
/// Can be used to implement any board transformation provided it only changes the position of
/// cells, not their values.
pub fn permute_board<const HIGH: u64, const LOW: u64>(board: u64) -> u64 {
    unsafe {
        let board = unpack_board(board);

        let mask = _mm_set_epi64x(HIGH as _, LOW as _);

        let shuffled = _mm_shuffle_epi8(board, mask);

        // Right shift cells that will go to high nibbles and add them to cells that will go to low
        // nibbles.
        // gapped_board contains cells packed 2 cells per byte, alternated with 0 bytes.
        let gapped_board = _mm_maddubs_epi16(shuffled, _mm_set1_epi16(0x1001));

        // Remove 0 bytes.
        let board = _mm_packus_epi16(gapped_board, gapped_board);

        _mm_cvtsi128_si64(board) as _
    }
}

pub fn transpose_board(board: u64) -> u64 {
    permute_board::<0x0f0b_0703_0e0a_0602, 0x0d09_0501_0c08_0400>(board)
}

pub fn transpose_rotate_board(board: u64) -> u64 {
    permute_board::<0x0004_080c_0105_090d, 0x0206_0a0e_0307_0b0f>(board)
}
