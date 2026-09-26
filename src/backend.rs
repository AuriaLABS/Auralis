//! Minimal backend/device boundary for Engine Phase A.
//!
//! Phase A deliberately covers one mature operation (matmul) and borrowed
//! host views only. No host/device transfer or allocator abstraction is hidden
//! behind this API.

use crate::kernels::{matmul_reference_into, matmul_row_slices_into};
use std::cell::{Cell, RefCell};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceId {
    Cpu,
    Mock(u32),
    Gpu,
}

impl DeviceId {
    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::Mock(index) => format!("mock:{index}"),
            Self::Gpu => "gpu".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendId {
    ScalarCpu,
    OptimizedCpu,
    Mock,
    Gpu,
}

impl BackendId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ScalarCpu => "cpu-scalar",
            Self::OptimizedCpu => "cpu-row-slices",
            Self::Mock => "mock-reference",
            Self::Gpu => "gpu-single",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    ZeroDimension {
        tensor: &'static str,
        rows: usize,
        cols: usize,
    },
    SizeOverflow {
        tensor: &'static str,
        rows: usize,
        cols: usize,
    },
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    InnerDimensionMismatch {
        a_cols: usize,
        b_rows: usize,
    },
    OutputShapeMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    DeviceMismatch {
        tensor: &'static str,
        backend: BackendId,
        expected: DeviceId,
        actual: DeviceId,
    },
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { tensor, rows, cols } => {
                write!(f, "{tensor} has zero dimension: {rows}x{cols}")
            }
            Self::SizeOverflow { tensor, rows, cols } => {
                write!(f, "{tensor} shape overflows usize: {rows}x{cols}")
            }
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "{tensor} length mismatch: expected {expected}, got {actual}"
            ),
            Self::InnerDimensionMismatch { a_cols, b_rows } => write!(
                f,
                "matmul inner dimension mismatch: A.cols={a_cols}, B.rows={b_rows}"
            ),
            Self::OutputShapeMismatch {
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "matmul output shape mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}"
            ),
            Self::DeviceMismatch {
                tensor,
                backend,
                expected,
                actual,
            } => write!(
                f,
                "{tensor} device mismatch for {}: expected {}, got {}",
                backend.as_str(),
                expected.label(),
                actual.label()
            ),
        }
    }
}

impl std::error::Error for BackendError {}

#[derive(Clone, Copy, Debug)]
pub struct MatrixRef<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
    device: DeviceId,
}

impl<'a> MatrixRef<'a> {
    pub fn new(
        data: &'a [f32],
        rows: usize,
        cols: usize,
        device: DeviceId,
    ) -> Result<Self, BackendError> {
        validate_storage("matrix", data.len(), rows, cols)?;
        Ok(Self {
            data,
            rows,
            cols,
            device,
        })
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn cols(self) -> usize {
        self.cols
    }

    pub fn device(self) -> DeviceId {
        self.device
    }

    pub fn as_slice(self) -> &'a [f32] {
        self.data
    }
}

#[derive(Debug)]
pub struct MatrixMut<'a> {
    data: &'a mut [f32],
    rows: usize,
    cols: usize,
    device: DeviceId,
}

impl<'a> MatrixMut<'a> {
    pub fn new(
        data: &'a mut [f32],
        rows: usize,
        cols: usize,
        device: DeviceId,
    ) -> Result<Self, BackendError> {
        validate_storage("output", data.len(), rows, cols)?;
        Ok(Self {
            data,
            rows,
            cols,
            device,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data
    }
}

fn validate_storage(
    tensor: &'static str,
    actual: usize,
    rows: usize,
    cols: usize,
) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 {
        return Err(BackendError::ZeroDimension { tensor, rows, cols });
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or(BackendError::SizeOverflow { tensor, rows, cols })?;
    if actual != expected {
        return Err(BackendError::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_matmul(
    backend: BackendId,
    device: DeviceId,
    a: MatrixRef<'_>,
    b: MatrixRef<'_>,
    out: &MatrixMut<'_>,
) -> Result<(), BackendError> {
    for (tensor, actual) in [("A", a.device), ("B", b.device), ("out", out.device)] {
        if actual != device {
            return Err(BackendError::DeviceMismatch {
                tensor,
                backend,
                expected: device,
                actual,
            });
        }
    }
    if a.cols != b.rows {
        return Err(BackendError::InnerDimensionMismatch {
            a_cols: a.cols,
            b_rows: b.rows,
        });
    }
    if out.rows != a.rows || out.cols != b.cols {
        return Err(BackendError::OutputShapeMismatch {
            expected_rows: a.rows,
            expected_cols: b.cols,
            actual_rows: out.rows,
            actual_cols: out.cols,
        });
    }
    Ok(())
}

pub trait Backend {
    fn id(&self) -> BackendId;
    fn device(&self) -> DeviceId;

    fn matmul(
        &self,
        a: MatrixRef<'_>,
        b: MatrixRef<'_>,
        mut out: MatrixMut<'_>,
    ) -> Result<(), BackendError> {
        validate_matmul(self.id(), self.device(), a, b, &out)?;
        self.matmul_validated(a, b, &mut out);
        Ok(())
    }

    fn matmul_validated(
        &self,
        a: MatrixRef<'_>,
        b: MatrixRef<'_>,
        out: &mut MatrixMut<'_>,
    );
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarCpuBackend;

impl Backend for ScalarCpuBackend {
    fn id(&self) -> BackendId {
        BackendId::ScalarCpu
    }

    fn device(&self) -> DeviceId {
        DeviceId::Cpu
    }

    fn matmul_validated(
        &self,
        a: MatrixRef<'_>,
        b: MatrixRef<'_>,
        out: &mut MatrixMut<'_>,
    ) {
        matmul_reference_into(
            a.data,
            a.rows,
            a.cols,
            b.data,
            b.cols,
            out.data,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizedCpuBackend;

impl Backend for OptimizedCpuBackend {
    fn id(&self) -> BackendId {
        BackendId::OptimizedCpu
    }

    fn device(&self) -> DeviceId {
        DeviceId::Cpu
    }

    fn matmul_validated(
        &self,
        a: MatrixRef<'_>,
        b: MatrixRef<'_>,
        out: &mut MatrixMut<'_>,
    ) {
        matmul_row_slices_into(
            a.data,
            a.rows,
            a.cols,
            b.data,
            b.cols,
            out.data,
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulCall {
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
}

#[derive(Debug)]
pub struct MockBackend {
    device: DeviceId,
    calls: Cell<usize>,
    trace: RefCell<Vec<MatmulCall>>,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new(0)
    }
}

impl MockBackend {
    pub fn new(device_index: u32) -> Self {
        Self {
            device: DeviceId::Mock(device_index),
            calls: Cell::new(0),
            trace: RefCell::new(Vec::new()),
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.get()
    }

    pub fn trace(&self) -> Vec<MatmulCall> {
        self.trace.borrow().clone()
    }
}

impl Backend for MockBackend {
    fn id(&self) -> BackendId {
        BackendId::Mock
    }

    fn device(&self) -> DeviceId {
        self.device
    }

    fn matmul_validated(
        &self,
        a: MatrixRef<'_>,
        b: MatrixRef<'_>,
        out: &mut MatrixMut<'_>,
    ) {
        self.calls.set(self.calls.get() + 1);
        self.trace.borrow_mut().push(MatmulCall {
            rows: a.rows,
            inner: a.cols,
            cols: b.cols,
        });
        matmul_reference_into(
            a.data,
            a.rows,
            a.cols,
            b.data,
            b.cols,
            out.data,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let raw = ((i * 41 + salt * 23 + i / 7) % 127) as f32;
                (raw - 63.0) / 37.0
            })
            .collect()
    }

    fn run_backend(
        backend: &dyn Backend,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Vec<f32> {
        let a = data(rows * inner, 3);
        let b = data(inner * cols, 11);
        let mut out = vec![f32::NAN; rows * cols];
        backend
            .matmul(
                MatrixRef::new(&a, rows, inner, backend.device()).unwrap(),
                MatrixRef::new(&b, inner, cols, backend.device()).unwrap(),
                MatrixMut::new(&mut out, rows, cols, backend.device()).unwrap(),
            )
            .unwrap();
        out
    }

    #[test]
    fn scalar_optimized_and_mock_are_exact_on_representative_shapes() {
        let scalar = ScalarCpuBackend;
        let optimized = OptimizedCpuBackend;
        let mock = MockBackend::new(7);
        for (rows, inner, cols) in [
            (1, 1, 1),
            (4, 3, 5),
            (9, 13, 37),
            (32, 32, 32),
            (32, 32, 96),
            (32, 96, 32),
            (32, 32, 100),
        ] {
            let reference = run_backend(&scalar, rows, inner, cols);
            assert_eq!(run_backend(&optimized, rows, inner, cols), reference);
            assert_eq!(run_backend(&mock, rows, inner, cols), reference);
        }
        assert_eq!(mock.call_count(), 7);
        assert_eq!(mock.trace()[2], MatmulCall { rows: 9, inner: 13, cols: 37 });
    }

    #[test]
    fn device_mismatch_fails_before_kernel_execution() {
        let backend = ScalarCpuBackend;
        let a = vec![1.0; 6];
        let b = vec![1.0; 6];
        let mut out = vec![0.0; 4];
        let err = backend
            .matmul(
                MatrixRef::new(&a, 2, 3, DeviceId::Mock(0)).unwrap(),
                MatrixRef::new(&b, 3, 2, DeviceId::Cpu).unwrap(),
                MatrixMut::new(&mut out, 2, 2, DeviceId::Cpu).unwrap(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            BackendError::DeviceMismatch { tensor: "A", .. }
        ));
    }

    #[test]
    fn shape_and_storage_errors_are_explicit() {
        assert!(matches!(
            MatrixRef::new(&[1.0; 3], 2, 2, DeviceId::Cpu),
            Err(BackendError::LengthMismatch { .. })
        ));
        assert!(matches!(
            MatrixRef::new(&[], 0, 2, DeviceId::Cpu),
            Err(BackendError::ZeroDimension { .. })
        ));

        let backend = OptimizedCpuBackend;
        let a = vec![1.0; 6];
        let b = vec![1.0; 8];
        let mut out = vec![0.0; 4];
        let err = backend
            .matmul(
                MatrixRef::new(&a, 2, 3, DeviceId::Cpu).unwrap(),
                MatrixRef::new(&b, 4, 2, DeviceId::Cpu).unwrap(),
                MatrixMut::new(&mut out, 2, 2, DeviceId::Cpu).unwrap(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            BackendError::InnerDimensionMismatch {
                a_cols: 3,
                b_rows: 4,
            }
        );
    }

    #[test]
    fn metadata_is_stable_for_benchmarks() {
        assert_eq!(BackendId::ScalarCpu.as_str(), "cpu-scalar");
        assert_eq!(BackendId::OptimizedCpu.as_str(), "cpu-row-slices");
        assert_eq!(BackendId::Mock.as_str(), "mock-reference");
        assert_eq!(BackendId::Gpu.as_str(), "gpu-single");
        assert_eq!(DeviceId::Cpu.label(), "cpu");
        assert_eq!(DeviceId::Mock(4).label(), "mock:4");
        assert_eq!(DeviceId::Gpu.label(), "gpu");
    }
}
