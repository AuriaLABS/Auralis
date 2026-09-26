//! #68 single-device GPU contract with mandatory CPU fallback.
//!
//! CI has no GPU. Hardware kernels stay unavailable. The product default
//! remains OptimizedCpu.

use crate::backend::{BackendId, DeviceId, MatrixMut, MatrixRef, ScalarCpuBackend};
use crate::device::{copy_to_device, copy_to_host, DeviceBuffer, DeviceError, DeviceProvenance};
use crate::backend::Backend;

pub const GPU_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    Unavailable,
    Device(DeviceError),
    Shape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuRuntime {
    pub available: bool,
}

impl GpuRuntime {
    pub fn detect() -> Self {
        Self { available: false }
    }

    pub fn provenance(&self) -> DeviceProvenance {
        DeviceProvenance {
            schema_version: crate::device::DEVICE_SCHEMA_VERSION,
            backend: BackendId::Gpu,
            device: DeviceId::Gpu,
        }
    }

    pub fn kernel_matmul(
        &self,
        _a: &DeviceBuffer,
        _b: &DeviceBuffer,
    ) -> Result<Vec<f32>, GpuError> {
        if !self.available {
            return Err(GpuError::Unavailable);
        }
        Err(GpuError::Unavailable)
    }

    pub fn fallback_matmul(
        &self,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let a_host = if a.device == DeviceId::Cpu {
            a.as_host().map_err(GpuError::Device)?.to_vec()
        } else {
            copy_to_host(a)
                .map_err(GpuError::Device)?
                .as_host()
                .map_err(GpuError::Device)?
                .to_vec()
        };
        let b_host = if b.device == DeviceId::Cpu {
            b.as_host().map_err(GpuError::Device)?.to_vec()
        } else {
            copy_to_host(b)
                .map_err(GpuError::Device)?
                .as_host()
                .map_err(GpuError::Device)?
                .to_vec()
        };
        if a_host.len() != rows * inner || b_host.len() != inner * cols {
            return Err(GpuError::Shape);
        }
        let mut out = vec![0.0; rows * cols];
        ScalarCpuBackend
            .matmul(
                MatrixRef::new(&a_host, rows, inner, DeviceId::Cpu).map_err(|_| GpuError::Shape)?,
                MatrixRef::new(&b_host, inner, cols, DeviceId::Cpu).map_err(|_| GpuError::Shape)?,
                MatrixMut::new(&mut out, rows, cols, DeviceId::Cpu).map_err(|_| GpuError::Shape)?,
            )
            .map_err(|_| GpuError::Shape)?;
        Ok(out)
    }

    pub fn staged_matmul(
        &self,
        a_host: &[f32],
        b_host: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let a = DeviceBuffer::on_host(a_host.to_vec()).map_err(GpuError::Device)?;
        let b = DeviceBuffer::on_host(b_host.to_vec()).map_err(GpuError::Device)?;
        let a_gpu = copy_to_device(&a, DeviceId::Gpu).map_err(GpuError::Device)?;
        let b_gpu = copy_to_device(&b, DeviceId::Gpu).map_err(GpuError::Device)?;
        match self.kernel_matmul(&a_gpu, &b_gpu) {
            Ok(out) => Ok(out),
            Err(GpuError::Unavailable) => self.fallback_matmul(&a_gpu, &b_gpu, rows, inner, cols),
            Err(err) => Err(err),
        }
    }
}

pub fn product_default_backend() -> BackendId {
    BackendId::OptimizedCpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_is_unavailable_and_not_the_default() {
        let gpu = GpuRuntime::detect();
        assert!(!gpu.available);
        assert_eq!(product_default_backend(), BackendId::OptimizedCpu);
        assert_eq!(gpu.provenance().label(), "gpu-single@gpu");
        let host = DeviceBuffer::on_host(vec![1.0, 2.0]).unwrap();
        let on_gpu = copy_to_device(&host, DeviceId::Gpu).unwrap();
        assert_eq!(
            gpu.kernel_matmul(&on_gpu, &on_gpu).unwrap_err(),
            GpuError::Unavailable
        );
    }

    #[test]
    fn fallback_matches_scalar_cpu() {
        let gpu = GpuRuntime::detect();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let via_gpu = gpu.staged_matmul(&a, &b, 2, 2, 2).unwrap();
        let mut direct = vec![0.0; 4];
        ScalarCpuBackend
            .matmul(
                MatrixRef::new(&a, 2, 2, DeviceId::Cpu).unwrap(),
                MatrixRef::new(&b, 2, 2, DeviceId::Cpu).unwrap(),
                MatrixMut::new(&mut direct, 2, 2, DeviceId::Cpu).unwrap(),
            )
            .unwrap();
        assert_eq!(via_gpu, direct);
    }
}
