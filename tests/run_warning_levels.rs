use highs::{HighsModelStatus, HighsSolutionStatus, Model, RowProblem, Sense};
use std::sync::Mutex;
use std::thread::ThreadId;

struct Capture {
    records: Mutex<Vec<(ThreadId, log::Level, String)>>,
}

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        self.records.lock().unwrap().push((
            std::thread::current().id(),
            record.level(),
            record.args().to_string(),
        ));
    }

    fn flush(&self) {}
}

static CAPTURE: Capture = Capture {
    records: Mutex::new(Vec::new()),
};

/// Solves `model` and returns every record this thread logged during the
/// solve; tests run on their own threads and share the global logger.
fn solve_capturing(model: Model) -> (highs::SolvedModel, Vec<(log::Level, String)>) {
    let _ = log::set_logger(&CAPTURE);
    log::set_max_level(log::LevelFilter::Trace);
    let me = std::thread::current().id();
    let before = CAPTURE.records.lock().unwrap().len();
    let solved = model.try_solve().expect("Highs_run must not error");
    let records = CAPTURE.records.lock().unwrap()[before..]
        .iter()
        .filter(|(thread, _, _)| *thread == me)
        .map(|(_, level, message)| (*level, message.clone()))
        .collect();
    (solved, records)
}

fn highs_run(records: &[(log::Level, String)]) -> Vec<(log::Level, String)> {
    records
        .iter()
        .filter(|(_, message)| message.contains("Highs_run"))
        .cloned()
        .collect()
}

fn weights(seed: u64, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        1.0 + ((state >> 33) % 97) as f64
    };
    (0..rows)
        .map(|_| (0..cols).map(|_| next()).collect())
        .collect()
}

const ROWS: usize = 6;
const COLS: usize = 60;

/// A market split whose rows have over/under slack, minimising the slack. All
/// binaries at zero with full under-slack is feasible, so the warm start is the
/// incumbent before the MIP solver starts its clock (`HighsMipSolver`'s
/// constructor), and the split is far too hard to prove.
fn slack_split_with_incumbent() -> Model {
    let w = weights(0x9E37_79B9_7F4A_7C15, ROWS, COLS);
    let mut problem = RowProblem::default();
    let items: Vec<_> = (0..COLS)
        .map(|_| problem.add_integer_column(0., 0..=1))
        .collect();
    let mut warm = vec![0.; COLS];
    for row in &w {
        let target = (row.iter().sum::<f64>() / 2.0).floor();
        let over = problem.add_column(1., 0..);
        let under = problem.add_column(1., 0..);
        let mut terms: Vec<_> = items.iter().copied().zip(row.iter().copied()).collect();
        terms.push((over, -1.));
        terms.push((under, 1.));
        problem.add_row(target..=target, terms);
        warm.push(0.);
        warm.push(target);
    }
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1);
    model.set_solution(Some(&warm), None, None, None);
    model
}

/// Equal-split rows over binaries with no slack: no trivial point is feasible,
/// so a solve stopped early has no incumbent.
fn hard_split_without_incumbent() -> Model {
    let w = weights(0x2545_F491_4F6C_DD1D, ROWS, COLS);
    let mut problem = RowProblem::default();
    let items: Vec<_> = (0..COLS)
        .map(|_| problem.add_integer_column(1., 0..=1))
        .collect();
    for row in &w {
        let target = (row.iter().sum::<f64>() / 2.0).floor();
        problem.add_row(
            target..=target,
            items.iter().copied().zip(row.iter().copied()),
        );
    }
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1);
    model
}

#[test]
fn time_limit_stop_with_an_incumbent_logs_highs_run_at_debug_only() {
    let mut model = slack_split_with_incumbent();
    model.set_option("time_limit", 0.0);

    let (solved, records) = solve_capturing(model);

    assert_eq!(solved.status(), HighsModelStatus::ReachedTimeLimit);
    assert_eq!(
        solved.primal_solution_status(),
        HighsSolutionStatus::Feasible
    );
    let run = highs_run(&records);
    assert_eq!(run.len(), 1, "{records:?}");
    assert_eq!(run[0].0, log::Level::Debug, "{run:?}");
    assert!(
        run[0].1.contains("model_status=ReachedTimeLimit"),
        "{run:?}"
    );
    assert!(
        records.iter().all(|(level, _)| *level > log::Level::Warn),
        "nothing at warn or above: {records:?}"
    );
}

#[test]
fn interrupt_with_an_incumbent_logs_highs_run_at_debug_only() {
    let mut model = slack_split_with_incumbent();
    model.set_mip_interrupt(|_| true);

    let (solved, records) = solve_capturing(model);

    assert_eq!(solved.status(), HighsModelStatus::ReachedInterrupt);
    assert_eq!(
        solved.primal_solution_status(),
        HighsSolutionStatus::Feasible
    );
    let run = highs_run(&records);
    assert_eq!(run.len(), 1, "{records:?}");
    assert_eq!(run[0].0, log::Level::Debug, "{run:?}");
    assert!(
        run[0].1.contains("model_status=ReachedInterrupt"),
        "{run:?}"
    );
}

#[test]
fn interrupt_without_an_incumbent_warns_with_status_and_reason() {
    let mut model = hard_split_without_incumbent();
    model.set_mip_interrupt(|_| true);

    let (solved, records) = solve_capturing(model);

    assert_eq!(solved.status(), HighsModelStatus::ReachedInterrupt);
    assert_ne!(
        solved.primal_solution_status(),
        HighsSolutionStatus::Feasible
    );
    let run = highs_run(&records);
    assert_eq!(run.len(), 1, "{records:?}");
    assert_eq!(run[0].0, log::Level::Warn, "{run:?}");
    assert!(
        run[0].1.contains("model_status=ReachedInterrupt"),
        "{run:?}"
    );
    assert!(run[0].1.contains("reason="), "{run:?}");
}

#[test]
fn time_limit_stop_without_an_incumbent_warns_with_status_and_reason() {
    let mut model = hard_split_without_incumbent();
    model.set_option("time_limit", 0.0);

    let (solved, records) = solve_capturing(model);

    assert_eq!(solved.status(), HighsModelStatus::ReachedTimeLimit);
    assert_ne!(
        solved.primal_solution_status(),
        HighsSolutionStatus::Feasible
    );
    let run = highs_run(&records);
    assert_eq!(run.len(), 1, "{records:?}");
    assert_eq!(run[0].0, log::Level::Warn, "{run:?}");
    assert!(
        run[0].1.contains("model_status=ReachedTimeLimit"),
        "{run:?}"
    );
    assert!(run[0].1.contains("reason="), "{run:?}");
}

#[test]
fn a_solve_that_finishes_logs_no_highs_run_line() {
    let mut problem = RowProblem::default();
    let x = problem.add_integer_column(1., 0..=10);
    let y = problem.add_integer_column(2., 0..=10);
    problem.add_row(3.., [(x, 1.), (y, 1.)]);
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();

    let (solved, records) = solve_capturing(model);

    assert_eq!(solved.status(), HighsModelStatus::Optimal);
    assert!(highs_run(&records).is_empty(), "{records:?}");
}
