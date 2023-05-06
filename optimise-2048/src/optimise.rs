use std::{
    io::{self, Write},
    mem::{self, MaybeUninit},
    ops::ControlFlow,
    ptr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use libc::{c_int, c_void};
use nalgebra::{ArrayStorage, Const, Matrix, SymmetricEigen, Vector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use lib_2048::{
    ai::{expectimax::ExpectimaxAi, Ai},
    control_flow_helper, logic, metrics, rng_seeds,
};

type MatrixN = Matrix<
    f64,
    Const<{ Weights::N }>,
    Const<{ Weights::N }>,
    ArrayStorage<f64, { Weights::N }, { Weights::N }>,
>;
type VectorN = Vector<f64, Const<{ Weights::N }>, ArrayStorage<f64, { Weights::N }, 1>>;

#[derive(Debug)]
pub struct Weights {
    pub mid_score_weight: f64,
    pub edge_score_weight: f64,
    pub corner_score_weight: f64,
    pub mid_empty_count_weight: f64,
    pub edge_empty_count_weight: f64,
    pub corner_empty_count_weight: f64,
    pub mid_merge_score_weight: f64,
    pub side_merge_score_weight: f64,
    pub edge_merge_score_weight: f64,
    pub corner_merge_score_weight: f64,
    pub mid_merge_count_weight: f64,
    pub side_merge_count_weight: f64,
    pub edge_merge_count_weight: f64,
    pub corner_merge_count_weight: f64,
    pub mid_monotonicity_score_weight: f64,
    pub edge_monotonicity_score_weight: f64,
    pub monotonicity_score_power: f64,
    pub loss_weight: f64,
}

impl Weights {
    const N: usize = 18;
}

impl From<VectorN> for Weights {
    fn from(value: VectorN) -> Self {
        let [[mid_score_weight, edge_score_weight, corner_score_weight, mid_empty_count_weight, edge_empty_count_weight, corner_empty_count_weight, mid_merge_score_weight, side_merge_score_weight, edge_merge_score_weight, corner_merge_score_weight, mid_merge_count_weight, side_merge_count_weight, edge_merge_count_weight, corner_merge_count_weight, mid_monotonicity_score_weight, edge_monotonicity_score_weight, monotonicity_score_power, loss_weight]] =
            value.data.0;

        Self {
            mid_score_weight,
            edge_score_weight,
            corner_score_weight,
            mid_empty_count_weight,
            edge_empty_count_weight,
            corner_empty_count_weight,
            mid_merge_score_weight,
            side_merge_score_weight,
            edge_merge_score_weight,
            corner_merge_score_weight,
            mid_merge_count_weight,
            side_merge_count_weight,
            edge_merge_count_weight,
            corner_merge_count_weight,
            mid_monotonicity_score_weight,
            edge_monotonicity_score_weight,
            monotonicity_score_power,
            loss_weight,
        }
    }
}

impl From<Weights> for VectorN {
    fn from(
        Weights {
            mid_score_weight,
            edge_score_weight,
            corner_score_weight,
            mid_empty_count_weight,
            edge_empty_count_weight,
            corner_empty_count_weight,
            mid_merge_score_weight,
            side_merge_score_weight,
            edge_merge_score_weight,
            corner_merge_score_weight,
            mid_merge_count_weight,
            side_merge_count_weight,
            edge_merge_count_weight,
            corner_merge_count_weight,
            mid_monotonicity_score_weight,
            edge_monotonicity_score_weight,
            monotonicity_score_power,
            loss_weight,
        }: Weights,
    ) -> Self {
        nalgebra::vector![
            mid_score_weight,
            edge_score_weight,
            corner_score_weight,
            mid_empty_count_weight,
            edge_empty_count_weight,
            corner_empty_count_weight,
            mid_merge_score_weight,
            side_merge_score_weight,
            edge_merge_score_weight,
            corner_merge_score_weight,
            mid_merge_count_weight,
            side_merge_count_weight,
            edge_merge_count_weight,
            corner_merge_count_weight,
            mid_monotonicity_score_weight,
            edge_monotonicity_score_weight,
            monotonicity_score_power,
            loss_weight
        ]
    }
}

static INTERRUPT_RECIEVED: AtomicBool = AtomicBool::new(false);

fn set_interrupt_handler() {
    extern "C" fn handle_interrupt(_: c_int, _: *mut libc::siginfo_t, _: *mut c_void) {
        INTERRUPT_RECIEVED.store(true, Ordering::Relaxed);
    }

    let handler: extern "C" fn(c_int, *mut libc::siginfo_t, *mut c_void) = handle_interrupt;

    let sa_mask = unsafe {
        let mut sa_mask = MaybeUninit::uninit();

        libc::sigemptyset(sa_mask.as_mut_ptr());

        sa_mask.assume_init()
    };

    let action = libc::sigaction {
        sa_sigaction: unsafe { mem::transmute(handler) },
        sa_mask,
        sa_flags: libc::SA_RESTART,
        sa_restorer: None,
    };

    unsafe {
        libc::sigaction(libc::SIGINT, &action, ptr::null_mut());
    }
}

fn board_merge_stats(mut f: impl FnMut(u16) -> (u32, u32), board: u64) -> (u32, u32, u32, u32) {
    let (mid_merge_stat, side_merge_stat) = [16, 32].into_iter().map(|i| (board >> i) as u16).fold(
        (0, 0),
        |(mid_merge_stat, side_merge_stat), row| {
            let (row_mid_merge_stat, row_side_merge_stat) = f(row);

            (
                mid_merge_stat + row_mid_merge_stat,
                side_merge_stat + row_side_merge_stat,
            )
        },
    );

    let (edge_merge_stat, corner_merge_stat) = [0, 48]
        .into_iter()
        .map(|i| (board >> i) as u16)
        .fold((0, 0), |(edge_merge_stat, corner_merge_stat), row| {
            let (row_edge_merge_stat, row_corner_merge_stat) = f(row);

            (
                edge_merge_stat + row_edge_merge_stat,
                corner_merge_stat + row_corner_merge_stat,
            )
        });

    (
        mid_merge_stat,
        side_merge_stat,
        edge_merge_stat,
        corner_merge_stat,
    )
}

// fn row_monotonicity_score(row: u16) -> u32 {
//     let score = row_cells(row)
//         .filter(|&cell| cell != 0)
//         .map(cell_score)
//         .iterator_array_windows()
//         .map(|[score1, score2]| {
//             if score1 > score2 {
//                 score1 as i32
//             } else if score1 < score2 {
//                 -(score2 as i32)
//             } else {
//                 0
//             }
//         })
//         .sum();

//     i32::abs(score) as _
// }

// fn row_monotonicity_score(row: u16) -> (u32, u32) {
//     let mut ascending_score = 0;
//     let mut descending_score = 0;

//     row_cells(row)
//         .filter(|&cell| cell != 0)
//         .map(cell_score)
//         .iterator_array_windows()
//         .for_each(|[score1, score2]| {
//             if score1 < score2 {
//                 ascending_score += score1 - score2;
//             } else if score1 > score2 {
//                 descending_score += score2 - score1;
//             }
//         });

//     (
//         cmp::max(ascending_score, descending_score),
//         cmp::min(ascending_score, descending_score),
//     )
// }

fn board_monotonicity_score(board: u64, power: f64) -> (f64, f64) {
    let mid_monotonicity_score = [16, 32]
        .into_iter()
        .map(|i| (board >> (i * 16)) as u16)
        .map(|row| metrics::row_monotonicity_score(row, power))
        .sum();

    let edge_monotonicity_score = [0, 48]
        .into_iter()
        .map(|i| (board >> (i * 16)) as u16)
        .map(|row| metrics::row_monotonicity_score(row, power))
        .sum();

    (mid_monotonicity_score, edge_monotonicity_score)
}

fn eval_board(
    mid_score_weight: f64,
    edge_score_weight: f64,
    corner_score_weight: f64,
    mid_empty_count_weight: f64,
    edge_empty_count_weight: f64,
    corner_empty_count_weight: f64,
    mid_merge_score_weight: f64,
    side_merge_score_weight: f64,
    edge_merge_score_weight: f64,
    corner_merge_score_weight: f64,
    mid_merge_count_weight: f64,
    side_merge_count_weight: f64,
    edge_merge_count_weight: f64,
    corner_merge_count_weight: f64,
    mid_monotonicity_score_weight: f64,
    edge_monotonicity_score_weight: f64,
    monotonicity_score_power: f64,
    board: u64,
) -> f64 {
    let mid_indices = [20, 24, 36, 40];
    let edge_indices = [4, 8, 16, 28, 32, 44, 52, 56];
    let corner_indices = [0, 12, 48, 60];

    let mid_score: u32 = mid_indices
        .into_iter()
        .map(|i| metrics::cell_score((board >> i) as u8 & 0xf))
        .sum();

    let edge_score: u32 = edge_indices
        .into_iter()
        .map(|i| metrics::cell_score((board >> i) as u8 & 0xf))
        .sum();

    let corner_score: u32 = corner_indices
        .into_iter()
        .map(|i| metrics::cell_score((board >> i) as u8 & 0xf))
        .sum();

    let mid_empty_count = mid_indices
        .into_iter()
        .filter(|i| (board >> i) & 0xf == 0)
        .count() as u32;

    let edge_empty_count = edge_indices
        .into_iter()
        .filter(|i| (board >> i) & 0xf == 0)
        .count() as u32;

    let corner_empty_count = corner_indices
        .into_iter()
        .filter(|i| (board >> i) & 0xf == 0)
        .count() as u32;

    let score = mid_score + edge_score + corner_score;

    let (mid_merge_score, side_merge_score, edge_merge_score, corner_merge_score) =
        [board, lib_2048::transpose_board(board)].into_iter().fold(
            (0, 0, 0, 0),
            |(
                total_mid_merge_score,
                total_side_merge_score,
                total_edge_merge_score,
                total_corner_merge_score,
            ),
             board| {
                let (mid_merge_score, side_merge_score, edge_merge_score, corner_merge_score) =
                    board_merge_stats(metrics::row_merge_scores, board);

                (
                    total_mid_merge_score + mid_merge_score,
                    total_side_merge_score + side_merge_score,
                    total_edge_merge_score + edge_merge_score,
                    total_corner_merge_score + corner_merge_score,
                )
            },
        );

    let (mid_merge_count, side_merge_count, edge_merge_count, corner_merge_count) =
        [board, lib_2048::transpose_board(board)].into_iter().fold(
            (0, 0, 0, 0),
            |(
                total_mid_merge_count,
                total_side_merge_count,
                total_edge_merge_count,
                total_corner_merge_count,
            ),
             board| {
                let (mid_merge_count, side_merge_count, edge_merge_count, corner_merge_count) =
                    board_merge_stats(metrics::row_merge_counts, board);

                (
                    total_mid_merge_count + mid_merge_count,
                    total_side_merge_count + side_merge_count,
                    total_edge_merge_count + edge_merge_count,
                    total_corner_merge_count + corner_merge_count,
                )
            },
        );

    let (mid_monotonicity_score, edge_monotonicity_score) =
        [board, lib_2048::transpose_board(board)].into_iter().fold(
            (0.0, 0.0),
            |(total_mid_monotonicity_score, total_edge_monotonicity_score), board| {
                let (mid_monotonicity_score, edge_monotonicity_score) =
                    board_monotonicity_score(board, monotonicity_score_power);

                (
                    total_mid_monotonicity_score + mid_monotonicity_score,
                    total_edge_monotonicity_score + edge_monotonicity_score,
                )
            },
        );

    let [mid_score, edge_score, corner_score, mid_empty_count, edge_empty_count, corner_empty_count, mid_merge_score, side_merge_score, edge_merge_score, corner_merge_score, mid_merge_count, side_merge_count, edge_merge_count, corner_merge_count] =
        [
            mid_score,
            edge_score,
            corner_score,
            mid_empty_count * score,
            edge_empty_count * score,
            corner_empty_count * score,
            mid_merge_score,
            side_merge_score,
            edge_merge_score,
            corner_merge_score,
            mid_merge_count * score,
            side_merge_count * score,
            edge_merge_count * score,
            corner_merge_count * score,
        ]
        .map(f64::from);

    [
        mid_score * mid_score_weight,
        edge_score * edge_score_weight,
        corner_score * corner_score_weight,
        mid_empty_count * mid_empty_count_weight,
        edge_empty_count * edge_empty_count_weight,
        corner_empty_count * corner_empty_count_weight,
        mid_merge_score * mid_merge_score_weight,
        side_merge_score * side_merge_score_weight,
        edge_merge_score * edge_merge_score_weight,
        corner_merge_score * corner_merge_score_weight,
        mid_merge_count * mid_merge_count_weight,
        side_merge_count * side_merge_count_weight,
        edge_merge_count * edge_merge_count_weight,
        corner_merge_count * corner_merge_count_weight,
        mid_monotonicity_score * mid_monotonicity_score_weight,
        edge_monotonicity_score * edge_monotonicity_score_weight,
    ]
    .into_iter()
    .sum()
}

fn eval_fitness(
    Weights {
        mid_score_weight,
        edge_score_weight,
        corner_score_weight,
        mid_empty_count_weight,
        edge_empty_count_weight,
        corner_empty_count_weight,
        mid_merge_score_weight,
        side_merge_score_weight,
        edge_merge_score_weight,
        corner_merge_score_weight,
        mid_merge_count_weight,
        side_merge_count_weight,
        edge_merge_count_weight,
        corner_merge_count_weight,
        mid_monotonicity_score_weight,
        edge_monotonicity_score_weight,
        monotonicity_score_power,
        loss_weight,
    }: Weights,
) -> f64 {
    let mut ai = ExpectimaxAi::new(1, loss_weight, |board| {
        eval_board(
            mid_score_weight,
            edge_score_weight,
            corner_score_weight,
            mid_empty_count_weight,
            edge_empty_count_weight,
            corner_empty_count_weight,
            mid_merge_score_weight,
            side_merge_score_weight,
            edge_merge_score_weight,
            corner_merge_score_weight,
            mid_merge_count_weight,
            side_merge_count_weight,
            edge_merge_count_weight,
            corner_merge_count_weight,
            mid_monotonicity_score_weight,
            edge_monotonicity_score_weight,
            monotonicity_score_power,
            board,
        )
    });

    rng_seeds::SEEDS
        .into_iter()
        .map(|seed| {
            let mut rng = ChaCha8Rng::from_seed(seed);

            let board = logic::spawn_square(&mut rng, 0);

            control_flow_helper::loop_try_fold((0, board), |(score, board)| {
                let maybe_direction = ai.get_next_move(board);

                maybe_direction.map_or(ControlFlow::Break(score), |direction| {
                    let new_board = logic::try_move(board, direction).unwrap().get();
                    let move_score = logic::eval_score(new_board) - logic::eval_score(board);
                    let new_board = logic::spawn_square(&mut rng, new_board);

                    ControlFlow::Continue((score + move_score, new_board))
                })
            })
        })
        .sum::<u32>()
        .into()
}

fn compute_mu_eff(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    let squared_sum: f64 = weights.iter().map(|&w| w * w).sum();

    sum * sum / squared_sum
}

pub fn optimise(
    thread_count: usize,
    per_thread: usize,
    mut sigma: f64,
    out: &mut impl Write,
) -> io::Result<()> {
    set_interrupt_handler();

    // Best score: 27041.12
    // Weights {
    //     mid_score_weight: -63.93960945949477,
    //     edge_score_weight: 29.455708558369096,
    //     corner_score_weight: 52.622632298042134,
    //     mid_empty_count_weight: -8.314031715335416,
    //     edge_empty_count_weight: 6.573888696369576,
    //     corner_empty_count_weight: 11.814890754842818,
    //     mid_merge_score_weight: -16.80298825778469,
    //     side_merge_score_weight: -13.811291747245749,
    //     edge_merge_score_weight: 5.238171106567343,
    //     corner_merge_score_weight: 8.818806897310912,
    //     mid_merge_count_weight: 23.659257198824623,
    //     side_merge_count_weight: 16.91773300470858,
    //     edge_merge_count_weight: 13.122227129739445,
    //     corner_merge_count_weight: 12.876169407381026,
    //     mid_monotonicity_score_weight: -11.581168176128301,
    //     edge_monotonicity_score_weight: 10.89609618616014,
    //     monotonicity_score_power: 0.21886156200338147,
    //     loss_weight: -4.8969563362373885,
    // }

    // let mut mean: VectorN = Weights {
    //     mid_score_weight: -63.93960945949477,
    //     edge_score_weight: 29.455708558369096,
    //     corner_score_weight: 52.622632298042134,
    //     mid_empty_count_weight: -8.314031715335416,
    //     edge_empty_count_weight: 6.573888696369576,
    //     corner_empty_count_weight: 11.814890754842818,
    //     mid_merge_score_weight: -16.80298825778469,
    //     side_merge_score_weight: -13.811291747245749,
    //     edge_merge_score_weight: 5.238171106567343,
    //     corner_merge_score_weight: 8.818806897310912,
    //     mid_merge_count_weight: 23.659257198824623,
    //     side_merge_count_weight: 16.91773300470858,
    //     edge_merge_count_weight: 13.122227129739445,
    //     corner_merge_count_weight: 12.876169407381026,
    //     mid_monotonicity_score_weight: -11.581168176128301,
    //     edge_monotonicity_score_weight: 10.89609618616014,
    //     monotonicity_score_power: 0.21886156200338147,
    //     loss_weight: -4.8969563362373885,
    // }
    // .into();

    let mut mean: VectorN = Weights {
        mid_score_weight: 0.0,
        edge_score_weight: 0.0,
        corner_score_weight: 0.0,
        mid_empty_count_weight: 0.0,
        edge_empty_count_weight: 0.0,
        corner_empty_count_weight: 0.0,
        mid_merge_score_weight: 0.0,
        side_merge_score_weight: 0.0,
        edge_merge_score_weight: 0.0,
        corner_merge_score_weight: 0.0,
        mid_merge_count_weight: 0.0,
        side_merge_count_weight: 0.0,
        edge_merge_count_weight: 0.0,
        corner_merge_count_weight: 0.0,
        mid_monotonicity_score_weight: 0.0,
        edge_monotonicity_score_weight: 0.0,
        monotonicity_score_power: 1.0,
        loss_weight: 0.0,
    }
    .into();

    let mut best: Option<(f64, Weights)> = None;

    let lambda = (thread_count * per_thread) as u32;
    let n = mean.nrows() as f64;

    let mut unscaled_weights: Vec<_> = (1..=lambda)
        .map(|i| f64::ln(f64::from(lambda + 1) * 0.5) - f64::ln(i.into()))
        .collect();

    let mu = unscaled_weights.iter().filter(|&&w| w >= 0.0).count();

    let mu_eff = compute_mu_eff(&unscaled_weights[0..mu]);

    let c_m = 1.0;
    let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
    let d_sigma = 1.0 + 2.0 * f64::max(0.0, f64::sqrt((mu_eff - 1.0) / (n + 1.0))) + c_sigma;

    let alpha_cov = 2.0;

    let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
    let c_1 = alpha_cov / ((n + 1.3) * (n + 1.3) + mu_eff);
    let c_mu = f64::min(
        1.0 - c_1,
        alpha_cov * (0.25 + mu_eff + 1.0 / mu_eff - 2.0)
            / ((n + 2.0) * (n + 2.0) + alpha_cov * mu_eff * 0.5),
    );

    let weights = {
        let mu_eff_neg = compute_mu_eff(&unscaled_weights[mu..]);

        let alpha_mu_neg = 1.0 + c_1 / c_mu;
        let alpha_mu_eff_neg = 1.0 + 2.0 * mu_eff_neg / (mu_eff + 2.0);
        let alpha_pos_def_neg = (1.0 - c_1 - c_mu) / (n + c_mu);

        let positive_sum: f64 = unscaled_weights[0..mu].iter().sum();

        let min_alpha = [alpha_mu_neg, alpha_mu_eff_neg, alpha_pos_def_neg]
            .into_iter()
            .min_by(f64::total_cmp)
            .unwrap();
        let negative_sum: f64 = unscaled_weights[mu..].iter().sum();

        unscaled_weights[0..mu]
            .iter_mut()
            .for_each(|w| *w /= positive_sum);
        unscaled_weights[mu..]
            .iter_mut()
            .for_each(|w| *w *= min_alpha / negative_sum);

        unscaled_weights
    };

    let mut p_sigma = VectorN::zeros();
    let mut p_c = VectorN::zeros();
    let mut covariance = MatrixN::identity();

    let mut rng = ChaCha8Rng::from_entropy();

    let mean_norm = f64::sqrt(2.0) * libm::tgamma(0.5 * (n + 1.0)) / libm::tgamma(0.5 * n);

    let mut g: u64 = 1;

    while !INTERRUPT_RECIEVED.load(Ordering::Relaxed) {
        let SymmetricEigen {
            mut eigenvalues,
            eigenvectors,
        } = covariance.clone().symmetric_eigen();
        eigenvalues.iter_mut().for_each(|x| *x = x.sqrt());
        let diagonal = MatrixN::from_diagonal(&eigenvalues);

        let mut samples: Vec<(f64, VectorN)> = {
            let raw_samples: Vec<_> = (0..lambda)
                .map(|_| {
                    let z = [(); Weights::N].map(|_| rng.sample(StandardNormal));

                    Vector::from_array_storage(ArrayStorage([z]))
                })
                .collect();

            let threads: Vec<_> = raw_samples
                .chunks_exact(per_thread)
                .map(Vec::from)
                .map(|array| {
                    thread::spawn(move || {
                        array
                            .iter()
                            .map(|z| {
                                let y = eigenvectors * diagonal * z;
                                let x = mean + sigma * y;

                                let fitness = eval_fitness(x.into());

                                (fitness, y)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();

            threads
                .into_iter()
                .flat_map(|thread| thread.join().unwrap())
                .collect()
        };

        samples
            .as_mut_slice()
            .sort_unstable_by(|(fitness1, _), (fitness2, _)| fitness2.total_cmp(fitness1));

        writeln!(
            out,
            "Iteration {g} (sigma {sigma}): Best score {}",
            samples[0].0 / rng_seeds::SEEDS.len() as f64
        )?;

        best = Some(if let Some(current_best) = best.take() {
            if current_best.0 >= samples[0].0 {
                current_best
            } else {
                (samples[0].0, (mean + sigma * samples[0].1).into())
            }
        } else {
            (samples[0].0, samples[0].1.into())
        });

        let samples: Vec<_> = samples.into_iter().map(|(_, y)| y).collect();

        let y_w: VectorN = samples
            .iter()
            .copied()
            .zip(weights[0..mu].iter().copied())
            .map(|(y, w)| y * w)
            .sum();

        mean += c_m * sigma * y_w;

        let covariance_inv_sqrt = {
            eigenvalues.iter_mut().for_each(|x| *x = 1.0 / *x);
            let inv_diagonal = Matrix::from_diagonal(&eigenvalues);

            eigenvectors * inv_diagonal * eigenvectors.transpose()
        };

        p_sigma = (1.0 - c_sigma) * p_sigma
            + f64::sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * covariance_inv_sqrt * y_w;

        sigma = sigma * f64::exp(c_sigma * (p_sigma.norm() / mean_norm - 1.0) / d_sigma);

        let h_sigma = if p_sigma.norm() / f64::sqrt(1.0 - (1.0 - c_sigma).powi(2 * (g as i32 + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * mean_norm
        {
            1.0
        } else {
            0.0
        };

        let delta_h_sigma = (1.0 - h_sigma) * c_c * (2.0 - c_c);

        p_c = (1.0 - c_c) * p_c + h_sigma * f64::sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w;

        let weight_degrees = {
            let negative_weights = weights[mu..]
                .iter()
                .copied()
                .zip(samples[mu..].iter().copied())
                .map(|(w, y)| w * n / (covariance_inv_sqrt * y).norm_squared());

            weights[0..mu].iter().copied().chain(negative_weights)
        };

        let weights_sum: f64 = weights.iter().copied().sum();

        let weight_degrees_term = c_mu
            * weight_degrees
                .zip(&samples)
                .map(|(w, &y)| w * y * y.transpose())
                .sum::<MatrixN>();

        covariance = (1.0 + c_1 * delta_h_sigma - c_1 - f64::from(c_mu * weights_sum)) * covariance
            + c_1 * p_c * p_c.transpose()
            + weight_degrees_term;

        g += 1;
    }

    if let Some((score, weights)) = best {
        write!(
            out,
            "Best score: {}\n{weights:#?}\n",
            score / rng_seeds::SEEDS.len() as f64
        )?;
    }

    Ok(())
}
