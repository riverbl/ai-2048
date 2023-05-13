use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
    mem,
    path::Path,
};

use core_2048::metrics;

const MID_SCORE_WEIGHT: f64 = -2.5308329994984358;
const EDGE_SCORE_WEIGHT: f64 = 9.28392368627789;
const CORNER_SCORE_WEIGHT: f64 = 6.068408079280982;
const MID_EMPTY_COUNT_WEIGHT: f64 = -0.21455536740379505;
const EDGE_EMPTY_COUNT_WEIGHT: f64 = 0.1605971954390412;
const CORNER_EMPTY_COUNT_WEIGHT: f64 = 0.034696325047842175;
const MID_MERGE_SCORE_WEIGHT: f64 = 1.3677044170294734;
const SIDE_MERGE_SCORE_WEIGHT: f64 = 1.835622259758967;
const EDGE_MERGE_SCORE_WEIGHT: f64 = -1.4738896477877101;
const CORNER_MERGE_SCORE_WEIGHT: f64 = 0.6213308535058433;
const MID_MERGE_COUNT_WEIGHT: f64 = 0.14212702916626757;
const SIDE_MERGE_COUNT_WEIGHT: f64 = 0.18624973329112077;
const EDGE_MERGE_COUNT_WEIGHT: f64 = 0.10935055309170347;
const CORNER_MERGE_COUNT_WEIGHT: f64 = 0.04694708224979276;
const MID_MONOTONICITY_SCORE_WEIGHT: f64 = -10.573744674677924;
const EDGE_MONOTONICITY_SCORE_WEIGHT: f64 = -6.314315836847477;

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
    let (mid_score, edge_score) = metrics::row_scores(row);
    let (mid_empty_count, edge_empty_count) = metrics::row_empty_counts(row);

    let (mid_merge_score, side_merge_score) = metrics::row_merge_scores(row);
    let (mid_merge_count, side_merge_count) = metrics::row_merge_counts(row);

    let monotonicity_score = metrics::row_monotonicity_score(row);

    let (mid_metric, edge_metric) = (
        f64::from(mid_score) * MID_SCORE_WEIGHT * 0.5
            + f64::from(edge_score) * EDGE_SCORE_WEIGHT * 0.5
            + f64::from(mid_merge_score) * MID_MERGE_SCORE_WEIGHT
            + f64::from(side_merge_score) * SIDE_MERGE_SCORE_WEIGHT
            + f64::from(monotonicity_score) * MID_MONOTONICITY_SCORE_WEIGHT,
        f64::from(mid_empty_count) * MID_EMPTY_COUNT_WEIGHT * 0.5
            + f64::from(edge_empty_count) * EDGE_EMPTY_COUNT_WEIGHT * 0.5
            + f64::from(mid_merge_count) * MID_MERGE_COUNT_WEIGHT
            + f64::from(side_merge_count) * SIDE_MERGE_COUNT_WEIGHT,
    );

    (mid_metric as _, edge_metric as _)
}

fn edge_row_metrics(row: u16) -> (f32, f32) {
    let (edge_score, corner_score) = metrics::row_scores(row);
    let (edge_empty_count, corner_empty_count) = metrics::row_empty_counts(row);

    let (edge_merge_score, corner_merge_score) = metrics::row_merge_scores(row);
    let (edge_merge_count, corner_merge_count) = metrics::row_merge_counts(row);

    let monotonicity_score = metrics::row_monotonicity_score(row);

    let (edge_metric, corner_metric) = (
        f64::from(edge_score) * EDGE_SCORE_WEIGHT * 0.5
            + f64::from(corner_score) * CORNER_SCORE_WEIGHT * 0.5
            + f64::from(edge_merge_score) * EDGE_MERGE_SCORE_WEIGHT
            + f64::from(corner_merge_score) * CORNER_MERGE_SCORE_WEIGHT
            + f64::from(monotonicity_score) * EDGE_MONOTONICITY_SCORE_WEIGHT,
        f64::from(edge_empty_count) * EDGE_EMPTY_COUNT_WEIGHT * 0.5
            + f64::from(corner_empty_count) * CORNER_EMPTY_COUNT_WEIGHT * 0.5
            + f64::from(edge_merge_count) * EDGE_MERGE_COUNT_WEIGHT
            + f64::from(corner_merge_count) * CORNER_MERGE_COUNT_WEIGHT,
    );

    (edge_metric as _, corner_metric as _)
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
    let rows = (0..=u16::MAX).map(move_row).map(|row| row.to_ne_bytes());

    write_table(file_path, rows)
}

fn write_score_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let scores = (0..=u8::MAX)
        .map(|cell_pair| {
            (0..2).fold(0, |score, i| -> u32 {
                let cell = u32::from(cell_pair >> (i * 4)) & 0xf;

                score + cell.saturating_sub(1) * (1 << cell)
            })
        })
        .map(|score| score.to_ne_bytes());

    write_table(file_path, scores)
}

fn write_turn_count_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let turn_counts = (0..=u8::MAX)
        .map(|cell_pair| {
            (0..2).fold(0, |turn_count, i| -> u16 {
                let cell = u16::from(cell_pair >> (i * 4)) & 0xf;

                turn_count + cell.checked_sub(1).map(|x| 1 << x).unwrap_or(0)
            })
        })
        .map(|turn_count| turn_count.to_ne_bytes());

    write_table(file_path, turn_counts)
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

    let turn_count_table_path = out_dir_path.join("turn_count_table");
    write_turn_count_table(&turn_count_table_path).unwrap();

    let mid_metrics_table_path = out_dir_path.join("mid_metrics_table");
    write_metrics_table(mid_row_metrics, &mid_metrics_table_path).unwrap();

    let edge_metrics_table_path = out_dir_path.join("edge_metrics_table");
    write_metrics_table(edge_row_metrics, &edge_metrics_table_path).unwrap();
}
