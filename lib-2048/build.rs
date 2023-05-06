use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
    mem,
    path::Path,
};

use core_2048::metrics;

const MID_SCORE_WEIGHT: f32 = 0.0;
const EDGE_SCORE_WEIGHT: f32 = 0.0;
const CORNER_SCORE_WEIGHT: f32 = 0.0;
const MID_EMPTY_COUNT_WEIGHT: f32 = 0.0;
const EDGE_EMPTY_COUNT_WEIGHT: f32 = 0.0;
const CORNER_EMPTY_COUNT_WEIGHT: f32 = 0.0;
const MID_MERGE_SCORE_WEIGHT: f32 = 0.0;
const SIDE_MERGE_SCORE_WEIGHT: f32 = 0.0;
const EDGE_MERGE_SCORE_WEIGHT: f32 = 0.0;
const CORNER_MERGE_SCORE_WEIGHT: f32 = 0.0;
const MID_MERGE_COUNT_WEIGHT: f32 = 0.0;
const SIDE_MERGE_COUNT_WEIGHT: f32 = 0.0;
const EDGE_MERGE_COUNT_WEIGHT: f32 = 0.0;
const CORNER_MERGE_COUNT_WEIGHT: f32 = 0.0;
const MID_MONOTONICITY_SCORE_WEIGHT: f32 = 0.0;
const EDGE_MONOTONICITY_SCORE_WEIGHT: f32 = 0.0;
const MONOTONICITY_SCORE_POWER: f64 = 1.0;

fn move_row(row: u16) -> u16 {
    let shift = row.trailing_zeros() & !0x3;

    let mut row = row.wrapping_shr(shift);

    let mut mask = 0;

    for j in [0, 4, 8] {
        mask = (mask << 4) | 0xf;

        let mut sub_row = row & !mask;
        row &= mask;
        let shift = sub_row.trailing_zeros() & !0x3;
        sub_row = sub_row.wrapping_shr(shift);

        let sub_row_cell = sub_row & 0xf;
        let row_cell = (row >> j) & 0xf;

        if sub_row_cell == row_cell && sub_row_cell != 0 {
            // If the cell is already 15 then incrementing it would overflow into the adjacent
            // cell.
            if row_cell != 15 {
                row += 1 << j;
            }
        } else if row_cell == 0 {
            row += sub_row_cell << j;
        } else {
            sub_row <<= 4;
        }

        row |= (sub_row << j) & !mask;
    }

    row
}

fn mid_row_metrics(row: u16) -> (f32, f32) {
    let mid_indices = [4, 8];
    let edge_indices = [0, 12];

    let mid_score: u32 = mid_indices
        .into_iter()
        .map(|i| metrics::cell_score((row >> i) as u8 & 0xf))
        .sum();

    let edge_score: u32 = edge_indices
        .into_iter()
        .map(|i| metrics::cell_score((row >> i) as u8 & 0xf))
        .sum();

    let mid_empty_count = mid_indices
        .into_iter()
        .filter(|i| (row >> i) & 0xf == 0)
        .count() as u32;

    let edge_empty_count = edge_indices
        .into_iter()
        .filter(|i| (row >> i) & 0xf == 0)
        .count() as u32;

    let (mid_merge_score, side_merge_score) = metrics::row_merge_scores(row);

    let (mid_merge_count, side_merge_count) = metrics::row_merge_counts(row);

    let monotonicity_score = metrics::row_monotonicity_score(row, MONOTONICITY_SCORE_POWER);

    (
        mid_score as f32 * MID_SCORE_WEIGHT * 0.5
            + edge_score as f32 * EDGE_SCORE_WEIGHT * 0.5
            + mid_merge_score as f32 * MID_MERGE_SCORE_WEIGHT
            + side_merge_score as f32 * SIDE_MERGE_SCORE_WEIGHT
            + monotonicity_score as f32 * MID_MONOTONICITY_SCORE_WEIGHT,
        mid_empty_count as f32 * MID_EMPTY_COUNT_WEIGHT * 0.5
            + edge_empty_count as f32 * EDGE_EMPTY_COUNT_WEIGHT * 0.5
            + mid_merge_count as f32 * MID_MERGE_COUNT_WEIGHT
            + side_merge_count as f32 * SIDE_MERGE_COUNT_WEIGHT,
    )
}

fn edge_row_metrics(row: u16) -> (f32, f32) {
    let edge_indices = [4, 8];
    let corner_indices = [0, 12];

    let edge_score: u32 = edge_indices
        .into_iter()
        .map(|i| metrics::cell_score((row >> i) as u8 & 0xf))
        .sum();

    let corner_score: u32 = edge_indices
        .into_iter()
        .map(|i| metrics::cell_score((row >> i) as u8 & 0xf))
        .sum();

    let edge_empty_count = edge_indices
        .into_iter()
        .filter(|i| (row >> i) & 0xf == 0)
        .count() as u32;

    let corner_empty_count = corner_indices
        .into_iter()
        .filter(|i| (row >> i) & 0xf == 0)
        .count() as u32;

    let (edge_merge_score, corner_merge_score) = metrics::row_merge_scores(row);

    let (edge_merge_count, corner_merge_count) = metrics::row_merge_counts(row);

    let monotonicity_score = metrics::row_monotonicity_score(row, MONOTONICITY_SCORE_POWER);

    (
        edge_score as f32 * EDGE_SCORE_WEIGHT * 0.5
            + corner_score as f32 * CORNER_SCORE_WEIGHT * 0.5
            + edge_merge_score as f32 * EDGE_MERGE_SCORE_WEIGHT
            + corner_merge_score as f32 * CORNER_MERGE_SCORE_WEIGHT
            + monotonicity_score as f32 * EDGE_MONOTONICITY_SCORE_WEIGHT,
        edge_empty_count as f32 * EDGE_EMPTY_COUNT_WEIGHT * 0.5
            + corner_empty_count as f32 * CORNER_EMPTY_COUNT_WEIGHT * 0.5
            + edge_merge_count as f32 * EDGE_MERGE_COUNT_WEIGHT
            + corner_merge_count as f32 * CORNER_MERGE_COUNT_WEIGHT,
    )
}

fn write_table<const N: usize>(
    file_path: &impl AsRef<Path>,
    items: impl IntoIterator<Item = [u8; N]>,
) -> io::Result<()> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    for item in items {
        writer.write_all(&item)?;
    }

    writer.flush()
}

fn write_move_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let rows = (0..=u16::MAX)
        .map(move_row)
        .map(|row| -> [u8; 2] { unsafe { mem::transmute(row) } });

    write_table(file_path, rows)
}

fn write_score_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let scores = (0..=u8::MAX)
        .map(|cell_pair| {
            (0..2).fold(0, |score, i| {
                let exponent = u32::from(cell_pair >> (i * 4)) & 0xf;

                score
                    + if exponent != 0 {
                        (exponent - 1) * (1 << exponent)
                    } else {
                        0
                    }
            })
        })
        .map(|score| -> [u8; 4] { unsafe { mem::transmute(score) } });

    write_table(file_path, scores)
}

fn write_metrics_table(
    f: impl FnMut(u16) -> (f32, f32),
    file_path: &impl AsRef<Path>,
) -> io::Result<()> {
    let metrics = (0..=u16::MAX)
        .map(f)
        .map(|metric| -> [u8; 8] { unsafe { mem::transmute(metric) } });

    write_table(file_path, metrics)
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir_path = Path::new(&out_dir);

    let move_table_path = out_dir_path.join("move_table");
    write_move_table(&move_table_path).unwrap();

    let score_table_path = out_dir_path.join("score_table");
    write_score_table(&score_table_path).unwrap();

    let mid_metrics_table_path = out_dir_path.join("mid_metrics_table");
    write_metrics_table(mid_row_metrics, &mid_metrics_table_path).unwrap();

    let edge_metrics_table_path = out_dir_path.join("edge_metrics_table");
    write_metrics_table(edge_row_metrics, &edge_metrics_table_path).unwrap();
}
