//! Error types for GPU operations.

use thiserror::Error;

/// Errors that can occur during GPU operations.
#[derive(Error, Debug)]
pub enum GpuError {
    /// Failed to initialize GPU adapter.
    #[error("Failed to request GPU adapter: no suitable adapter found")]
    AdapterNotFound,

    /// Failed to create GPU device.
    #[error("Failed to create GPU device: {0}")]
    DeviceCreation(#[from] wgpu::RequestDeviceError),

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    /// Invalid parameters.
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    /// GPU execution failed.
    #[error("GPU execution failed: {0}")]
    ExecutionFailed(String),

    /// Buffer mapping failed.
    #[error("Buffer async mapping failed: {0}")]
    BufferMapping(#[from] wgpu::BufferAsyncError),
}
