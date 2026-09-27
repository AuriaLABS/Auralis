//! #68 single-device GPU contract with mandatory CPU fallback.
//!
//! CI has no GPU. Hardware kernels stay unavailable. The product default
//! remains OptimizedCpu. Staging copies are explicit and counted.
//! Engine `DeviceId` / `BackendId` stay unchanged so model-backend
//! Phase B is not in this slice.

use crate::backend::{
    Backend, BackendId, DeviceId, MatrixMut, MatrixRef, OptimizedCpuBackend,
};
use crate::device::{copy_to_device, copy_to_host, DeviceBuffer, DeviceError};

pub const GPU_SCHEMA_VERSION: u32 = 1;
pub const GPU_STAGING_DEVICE: DeviceId = DeviceId::Mock(u32::MAX);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    Unavailable,
    Device(DeviceError),
    Shape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferCost {
    pub host_to_device: usize,
    pub device_to_host: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuRuntime {
    available: bool,
}

impl GpuRuntime {
    pub fn detect() -> Self {
        Self { available: false }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn provenance_label(&self) -> &'static str {
        "gpu-single@gpu"
    }

    pub fn kernel_matmul(
        &self,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
    ) -> Result<Vec<f32>, GpuError> {
        if a.device != GPU_STAGING_DEVICE || b.device != GPU_STAGING_DEVICE {
            return Err(GpuError::Device(DeviceError::DeviceMismatch));
        }
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
    ) -> Result<(Vec<f32>, TransferCost), GpuError> {
        let mut cost = TransferCost {
            host_to_device: 0,
            device_to_host: 0,
        };
        let a_host = host_view(a, &mut cost)?;
        let b_host = host_view(b, &mut cost)?;
        if a_host.len() != rows * inner || b_host.len() != inner * cols {
            return Err(GpuError::Shape);
        }
        let mut out = vec![0.0; rows * cols];
        OptimizedCpuBackend
            .matmul(
                MatrixRef::new(&a_host, rows, inner, DeviceId::Cpu).map_err(|_| GpuError::Shape)?,
                MatrixRef::new(&b_host, inner, cols, DeviceId::Cpu).map_err(|_| GpuError::Shape)?,
                MatrixMut::new(&mut out, rows, cols, DeviceId::Cpu).map_err(|_| GpuError::Shape)?,
            )
            .map_err(|_| GpuError::Shape)?;
        Ok((out, cost))
    }

    pub fn staged_matmul(
        &self,
        a_host: &[f32],
        b_host: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, TransferCost), GpuError> {
        let a = DeviceBuffer::on_host(a_host.to_vec()).map_err(GpuError::Device)?;
        let b = DeviceBuffer::on_host(b_host.to_vec()).map_err(GpuError::Device)?;
        let a_gpu = copy_to_device(&a, GPU_STAGING_DEVICE).map_err(GpuError::Device)?;
        let b_gpu = copy_to_device(&b, GPU_STAGING_DEVICE).map_err(GpuError::Device)?;
        match self.kernel_matmul(&a_gpu, &b_gpu) {
            Ok(out) => Ok((
                out,
                TransferCost {
                    host_to_device: a_host.len() + b_host.len(),
                    device_to_host: rows * cols,
                },
            )),
            Err(GpuError::Unavailable) => {
                let (out, mut cost) = self.fallback_matmul(&a_gpu, &b_gpu, rows, inner, cols)?;
                cost.host_to_device += a_host.len() + b_host.len();
                Ok((out, cost))
            }
            Err(err) => Err(err),
        }
    }
}

fn host_view(buf: &DeviceBuffer, cost: &mut TransferCost) -> Result<Vec<f32>, GpuError> {
    if buf.device == DeviceId::Cpu {
        return buf
            .as_host()
            .map(|s| s.to_vec())
            .map_err(GpuError::Device);
    }
    cost.device_to_host += buf.len();
    copy_to_host(buf)
        .map_err(GpuError::Device)?
        .as_host()
        .map(|s| s.to_vec())
        .map_err(GpuError::Device)
}

pub fn product_default_backend() -> BackendId {
    BackendId::OptimizedCpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ScalarCpuBackend;

    #[test]
    fn gpu_is_unavailable_and_not_the_default() {
        let gpu = GpuRuntime::detect();
        assert!(!gpu.is_available());
        assert_eq!(product_default_backend(), BackendId::OptimizedCpu);
        assert_eq!(gpu.provenance_label(), "gpu-single@gpu");
        let host = DeviceBuffer::on_host(vec![1.0, 2.0]).unwrap();
        assert_eq!(
            gpu.kernel_matmul(&host, &host).unwrap_err(),
            GpuError::Device(DeviceError::DeviceMismatch)
        );
        let on_gpu = copy_to_device(&host, GPU_STAGING_DEVICE).unwrap();
        assert_eq!(
            gpu.kernel_matmul(&on_gpu, &on_gpu).unwrap_err(),
            GpuError::Unavailable
        );
    }

    #[test]
    fn fallback_matches_cpu_and_counts_transfers() {
        let gpu = GpuRuntime::detect();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let (via_gpu, cost) = gpu.staged_matmul(&a, &b, 2, 2, 2).unwrap();
        let mut direct = vec![0.0; 4];
        ScalarCpuBackend
            .matmul(
                MatrixRef::new(&a, 2, 2, DeviceId::Cpu).unwrap(),
                MatrixRef::new(&b, 2, 2, DeviceId::Cpu).unwrap(),
                MatrixMut::new(&mut direct, 2, 2, DeviceId::Cpu).unwrap(),
            )
            .unwrap();
        assert_eq!(via_gpu, direct);
        assert_eq!(cost.host_to_device, 8);
        assert_eq!(cost.device_to_host, 8);
    }
}
