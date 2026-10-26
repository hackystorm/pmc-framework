// SPDX-License-Identifier: GPL-3.0-only

//! Adam optimiser state.

use pmc_core::Tensor;

pub struct AdamState {
    pub first_moment: Vec<Tensor>,
    pub second_moment: Vec<Tensor>,
    pub step: u32,
}

impl AdamState {
    pub fn new(shapes: &[Vec<usize>]) -> AdamState {
        AdamState {
            first_moment: shapes.iter().map(|shape| Tensor::zeros(shape)).collect(),
            second_moment: shapes.iter().map(|shape| Tensor::zeros(shape)).collect(),
            step: 0,
        }
    }

    pub fn step(
        &mut self,
        params: &mut [&mut Tensor],
        grads: &[&Tensor],
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) {
        assert_eq!(params.len(), grads.len(), "parameter and gradient count differ");
        assert_eq!(params.len(), self.first_moment.len(), "optimiser state mismatch");

        self.step += 1;
        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        for index in 0..params.len() {
            let param = &mut params[index];
            let grad = &grads[index];
            assert_eq!(
                param.data.len(),
                grad.data.len(),
                "gradient shape does not match the parameter"
            );
            let m = &mut self.first_moment[index];
            let v = &mut self.second_moment[index];

            for i in 0..param.data.len() {
                m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * grad.data[i];
                v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * grad.data[i] * grad.data[i];
                let m_hat = m.data[i] / bias_correction1;
                let v_hat = v.data[i] / bias_correction2;
                param.data[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }
        }
    }
}
