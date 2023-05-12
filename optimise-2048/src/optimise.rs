use std::{
    io::{self, Write},
    ops::ControlFlow,
    thread,
};

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

#[derive(Clone, Copy, Debug)]
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
    pub loss_weight: f64,
}

impl Weights {
    const N: usize = 17;
}

impl From<VectorN> for Weights {
    fn from(value: VectorN) -> Self {
        let [[mid_score_weight, edge_score_weight, corner_score_weight, mid_empty_count_weight, edge_empty_count_weight, corner_empty_count_weight, mid_merge_score_weight, side_merge_score_weight, edge_merge_score_weight, corner_merge_score_weight, mid_merge_count_weight, side_merge_count_weight, edge_merge_count_weight, corner_merge_count_weight, mid_monotonicity_score_weight, edge_monotonicity_score_weight, loss_weight]] =
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
            loss_weight
        ]
    }
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
    board: u64,
) -> f64 {
    let mid_row_metrics = |row| -> (f32, f32) {
        let (mid_score, edge_score) = metrics::row_scores(row);
        let (mid_empty_count, edge_empty_count) = metrics::row_empty_counts(row);

        let (mid_merge_score, side_merge_score) = metrics::row_merge_scores(row);
        let (mid_merge_count, side_merge_count) = metrics::row_merge_counts(row);

        let monotonicity_score = metrics::row_monotonicity_score(row);

        let (mid_metric, edge_metric) = (
            f64::from(mid_score) * mid_score_weight * 0.5
                + f64::from(edge_score) * edge_score_weight * 0.5
                + f64::from(mid_merge_score) * mid_merge_score_weight
                + f64::from(side_merge_score) * side_merge_score_weight
                + f64::from(monotonicity_score) * mid_monotonicity_score_weight,
            f64::from(mid_empty_count) * mid_empty_count_weight * 0.5
                + f64::from(edge_empty_count) * edge_empty_count_weight * 0.5
                + f64::from(mid_merge_count) * mid_merge_count_weight
                + f64::from(side_merge_count) * side_merge_count_weight,
        );

        (mid_metric as _, edge_metric as _)
    };

    let edge_row_metrics = |row| -> (f32, f32) {
        let (edge_score, corner_score) = metrics::row_scores(row);
        let (edge_empty_count, corner_empty_count) = metrics::row_empty_counts(row);

        let (edge_merge_score, corner_merge_score) = metrics::row_merge_scores(row);
        let (edge_merge_count, corner_merge_count) = metrics::row_merge_counts(row);

        let monotonicity_score = metrics::row_monotonicity_score(row);

        let (edge_metric, corner_metric) = (
            f64::from(edge_score) * edge_score_weight * 0.5
                + f64::from(corner_score) * corner_score_weight * 0.5
                + f64::from(edge_merge_score) * edge_merge_score_weight
                + f64::from(corner_merge_score) * corner_merge_score_weight
                + f64::from(monotonicity_score) * edge_monotonicity_score_weight,
            f64::from(edge_empty_count) * edge_empty_count_weight * 0.5
                + f64::from(corner_empty_count) * corner_empty_count_weight * 0.5
                + f64::from(edge_merge_count) * edge_merge_count_weight
                + f64::from(corner_merge_count) * corner_merge_count_weight,
        );

        (edge_metric as _, corner_metric as _)
    };

    // scale is expected to grow with number of turns roughly as fast as score.
    let scale = {
        let turn_count: f64 = logic::count_turns(board).into();
        f64::ln(turn_count) * turn_count
    };

    [board, lib_2048::transpose_board(board)]
        .into_iter()
        .fold(0.0, |total_metrics, board| {
            let mid_metrics: f64 = [16, 32]
                .into_iter()
                .map(|i| {
                    let row = (board >> i) as u16;
                    let (score_metrics, count_metrics) = mid_row_metrics(row);

                    f64::from(count_metrics).mul_add(scale, score_metrics.into())
                })
                .sum();

            let edge_metrics: f64 = [0, 48]
                .into_iter()
                .map(|i| {
                    let row = (board >> i) as u16;
                    let (score_metrics, count_metrics) = edge_row_metrics(row);

                    f64::from(count_metrics).mul_add(scale, score_metrics.into())
                })
                .sum();

            total_metrics + mid_metrics + edge_metrics
        })
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
            board,
        )
    });

    rng_seeds::SEEDS
        .into_iter()
        .map(|seed| {
            let mut rng = ChaCha8Rng::from_seed(seed);

            let board = logic::get_initial_board(&mut rng);

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

fn write_iteration_info(
    out: &mut impl Write,
    overall_best_score: f64,
    overall_best_weights: Weights,
    iteration_score: f64,
    iteration_sigma: f64,
    g: u64,
) -> io::Result<()> {
    // Move cursor to start of current line and clear current line.
    out.write(b"\r\x1b[K")?;

    if g > 1 {
        // If this is not the first iteration, delete the previously printed best weights.
        (0..(Weights::N + 3)).try_for_each(|_| out.write(b"\x1b[F\x1b[K").map(|_| ()))?;
    }

    writeln!(
        out,
        "Iteration {g} (sigma {iteration_sigma}): Best score {iteration_score}"
    )?;

    write!(
        out,
        "Best overall score: {overall_best_score}\n{overall_best_weights:#?}\n"
    )?;

    out.flush()
}

pub fn optimise(
    thread_count: usize,
    per_thread: usize,
    mut sigma: f64,
    out: &mut impl Write,
) -> io::Result<!> {
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
        loss_weight: 0.0,
    }
    .into();

    let mut overall_best: Option<(f64, Weights)> = None;

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

    loop {
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

        overall_best = {
            let iteration_best_score = samples[0].0 / rng_seeds::SEEDS.len() as f64;

            let (new_best_score, new_best_weights) = overall_best
                .take()
                .filter(|&(current_best_score, _)| current_best_score >= iteration_best_score)
                .unwrap_or_else(|| (iteration_best_score, (mean + sigma * samples[0].1).into()));

            write_iteration_info(
                out,
                new_best_score,
                new_best_weights,
                iteration_best_score,
                sigma,
                g,
            )?;

            Some((new_best_score, new_best_weights))
        };

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
}
