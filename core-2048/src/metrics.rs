use std::cmp;

pub fn cell_score(cell: u8) -> u32 {
    u32::from(cell.saturating_sub(1)) * (1 << cell)
}

pub fn row_merge_counts(row: u16) -> (u32, u32) {
    let cells = [0, 4, 8, 12].map(|i| (row >> i) & 0xf);

    let mid_merge_count = (cells[1] == cells[2]).into();

    let side_merge_count = [[0, 1], [2, 3]]
        .into_iter()
        .filter(|&[i, j]| cells[i] == cells[j])
        .count() as _;

    (mid_merge_count, side_merge_count)
}

pub fn row_merge_scores(row: u16) -> (u32, u32) {
    let merge_scores = [0, 4, 8, 12]
        .map(|i| (row >> i) & 0xf)
        .map(|cell| 1 << (cell + 1));

    let mid_merge_score = if merge_scores[1] == merge_scores[2] {
        merge_scores[1]
    } else {
        0
    };

    let side_merge_score = [[0, 1], [2, 3]]
        .into_iter()
        .filter(|&[i, j]| merge_scores[i] == merge_scores[j])
        .map(|[i, _]| merge_scores[i])
        .sum();

    (mid_merge_score, side_merge_score)
}

pub fn row_monotonicity_score(row: u16, power: f64) -> f64 {
    let scores = [0, 4, 8, 12]
        .map(|i| (row >> i) as u8 & 0xf)
        .map(cell_score);

    let (increasing_score, decreasing_score) = scores
        .map(|score| f64::powf(score.into(), power))
        .array_windows()
        .fold(
            (0.0, 0.0),
            |(increasing_score, decreasing_score), [num1, num2]| {
                if num1 > num2 {
                    (increasing_score, decreasing_score + num1 - num2)
                } else {
                    (increasing_score + num2 - num1, decreasing_score)
                }
            },
        );

    cmp::min_by(increasing_score, decreasing_score, f64::total_cmp)
}
