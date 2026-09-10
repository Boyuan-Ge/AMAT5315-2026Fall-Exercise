use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    #[serde(rename = "L")]
    pub l: usize,
    #[serde(rename = "T")]
    pub t: f64,
    pub sweep: u64,
    pub m: f64,
    pub spins: String,
}
