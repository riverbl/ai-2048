#![feature(const_transmute_copy, maybe_uninit_slice, maybe_uninit_uninit_array)]

pub use core_2048::*;

pub mod ai;
pub mod control_flow_helper;
pub mod logic;
pub mod rng_seeds;

mod direction;
