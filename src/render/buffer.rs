use crate::render::{resources::Buffer, util::ToErupt};
use erupt::vk;
use gpu_alloc::UsageFlags;
use std::num::NonZeroU64;

pub struct BufferInfo {
    pub align: u64,
    pub size: u64,
    pub usage_flags: vk::BufferUsageFlags,
    pub allocation_flags: UsageFlags,
}

#[derive(Clone)]
pub struct BufferRegion {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
    pub stride: Option<u64>,
}

impl BufferRegion {
    pub fn whole(buffer: Buffer) -> Self {
        BufferRegion {
            offset: 0,
            size: buffer.info().size,
            buffer,
            stride: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceAddress(pub NonZeroU64);

impl DeviceAddress {
    pub fn new(address: u64) -> DeviceAddress {
        NonZeroU64::new(address).map(DeviceAddress).unwrap()
    }

    pub fn offset(&mut self, offset: u64) -> DeviceAddress {
        let value = self.0.get().checked_add(offset).unwrap();
        DeviceAddress(unsafe { NonZeroU64::new_unchecked(value) })
    }
}

impl ToErupt<vk::DeviceOrHostAddressKHR> for DeviceAddress {
    fn to_erupt(&self) -> vk::DeviceOrHostAddressKHR {
        vk::DeviceOrHostAddressKHR {
            device_address: self.0.get(),
        }
    }
}

impl ToErupt<vk::DeviceOrHostAddressConstKHR> for DeviceAddress {
    fn to_erupt(&self) -> vk::DeviceOrHostAddressConstKHR {
        vk::DeviceOrHostAddressConstKHR {
            device_address: self.0.get(),
        }
    }
}
