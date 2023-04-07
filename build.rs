#![feature(is_sorted)]

use std::{
    env,
    fmt::Display,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

fn move_row(row: u16) -> u16 {
    let shift = row.trailing_zeros() & !0x3;

    let mut row = row.wrapping_shr(shift);

    let mut mask = 0;

    for j in [0, 4] {
        mask = (mask << 4) | 0xf;

        let mut sub_row = row & !mask;
        row &= mask;
        let shift = sub_row.trailing_zeros() & !0x3;
        sub_row = sub_row.wrapping_shr(shift);

        if sub_row & 0xf == (row >> j) & 0xf && sub_row & 0xf != 0 {
            // This can overflow into if the adjacent square if the square being incremented
            // has reached 15.
            row += 1 << j;
        } else {
            sub_row <<= 4;
        }

        row |= (sub_row << j) & !mask;
    }

    if row >> 12 == (row >> 8) & 0xf && row >> 12 != 0 {
        // This can overflow into if the adjacent square if the square being incremented
        // has reached 15.
        row &= 0xfff;
        row += 1 << 8;
    }

    row
}

fn row_metrics(row: u16) -> i8 {
    let merge_count = (0..3)
        .filter(|i| {
            let cell1 = (row >> (i * 4)) & 0xf;
            let cell2 = (row >> (i * 4 + 4)) & 0xf;

            cell1 == 0 || cell2 == 0 || cell1 == cell2
        })
        .count();

    let monitonic_score = {
        let mut cells = [0u8; 4];

        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = ((row >> (i * 4)) & 0xf) as _;
        }

        let is_monotonic = cells.iter().is_sorted() || cells.iter().rev().is_sorted();

        cells.sort_unstable();

        let score: i8 = cells[0..3]
            .iter()
            .map(|&cell| cell.saturating_sub(6))
            .map(|x| (x * x / 3) as i8)
            .sum();

        if is_monotonic {
            score
        } else {
            -score
        }
    };

    merge_count as i8 + monitonic_score
}

fn write_table(
    file_path: &impl AsRef<Path>,
    items: impl IntoIterator<Item = impl Display>,
) -> io::Result<()> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(b"#[allow(clippy::unreadable_literal)]\n[")?;

    for item in items {
        write!(writer, "{item},")?;
    }

    writer.write_all(b"]\n")?;
    writer.flush()
}

fn write_move_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let rows = (0..=u16::MAX).map(move_row);

    write_table(file_path, rows)
}

fn write_score_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let scores = (0..=u8::MAX).map(|cell_pair| {
        (0..2).fold(0, |score, i| {
            let exponent = u32::from(cell_pair >> (i * 4)) & 0xf;

            score
                + if exponent != 0 {
                    (exponent - 1) * (1 << exponent)
                } else {
                    0
                }
        })
    });

    write_table(file_path, scores)
}

fn write_metrics_table(file_path: &impl AsRef<Path>) -> io::Result<()> {
    let metrics = (0..=u16::MAX).map(row_metrics);

    write_table(file_path, metrics)
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir_path = Path::new(&out_dir);

    let move_table_path = out_dir_path.join("move_table.rs");
    write_move_table(&move_table_path).unwrap();

    let score_table_path = out_dir_path.join("score_table.rs");
    write_score_table(&score_table_path).unwrap();

    let metrics_table_path = out_dir_path.join("metrics_table.rs");
    write_metrics_table(&metrics_table_path).unwrap();
}
