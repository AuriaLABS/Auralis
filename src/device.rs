//! #67 Scale device storage and explicit host↔device copies.
//!
//! Engine `backend.rs` stays the CPU kernel boundary. This module owns
//! buffers and forbids implicit host views of device memory.

use crate::backend::{BackendId, DeviceId};

pub const DEVICE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    Empty,
    DeviceMismatch,
    ImplicitCopy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceBuffer {
    pub device: DeviceId,
    data: Vec<f32>,
}

impl DeviceBuffer {
    pub fn on_host(data: Vec<f32>) -> Result<Self, DeviceError> {
        if data.is_empty() {
            return Err(DeviceError::Empty);
        }
        Ok(Self {
            device: DeviceId::Cpu,
            data,
        })
    }

    pub fn as_host(&self) -> Result<&[f32], DeviceError> {
        if self.device != DeviceId::Cpu {
            return Err(DeviceError::ImplicitCopy);
        }
        Ok(&self.data)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub fn copy_to_device(host: &DeviceBuffer, device: DeviceId) -> Result<DeviceBuffer, DeviceError> {
    if host.device != DeviceId::Cpu {
        return Err(DeviceError::DeviceMismatch);
    }
    Ok(DeviceBuffer {
        device,
        data: host.data.clone(),
    })
}

pub fn copy_to_host(device: &DeviceBuffer) -> Result<DeviceBuffer, DeviceError> {
    if device.device == DeviceId::Cpu {
        return Err(DeviceError::DeviceMismatch);
    }
    Ok(DeviceBuffer {
        device: DeviceId::Cpu,
        data: device.data.clone(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProvenance {
    pub schema_version: u32,
    pub backend: BackendId,
    pub device: DeviceId,
}

impl DeviceProvenance {
    pub fn cpu_default() -> Self {
        Self {
            schema_version: DEVICE_SCHEMA_VERSION,
            backend: BackendId::OptimizedCpu,
            device: DeviceId::Cpu,
        }
    }

    pub fn label(&self) -> String {
        format!("{}@{}", self.backend.as_str(), self.device.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_baseline_roundtrip_is_explicit() {
        let host = DeviceBuffer::on_host(vec![1.0, 2.0, 3.0]).unwrap();
        let device = copy_to_device(&host, DeviceId::Mock(1)).unwrap();
        let back = copy_to_host(&device).unwrap();
        assert_eq!(host.as_host().unwrap(), back.as_host().unwrap());
        assert_eq!(device.device, DeviceId::Mock(1));
        assert_eq!(DeviceProvenance::cpu_default().label(), "cpu-row-slices@cpu");
    }

    #[test]
    fn implicit_host_view_of_device_fails() {
        let host = DeviceBuffer::on_host(vec![1.0]).unwrap();
        let device = copy_to_device(&host, DeviceId::Mock(0)).unwrap();
        assert_eq!(device.as_host().unwrap_err(), DeviceError::ImplicitCopy);
        assert_eq!(
            copy_to_device(&device, DeviceId::Mock(1)).unwrap_err(),
            DeviceError::DeviceMismatch
        );
        assert_eq!(DeviceBuffer::on_host(vec![]).unwrap_err(), DeviceError::Empty);
    }
}
