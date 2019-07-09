//use crate::ComplexUnitary;
//use crate::circuits::{GateKronecker, GateProduct, GateSingleQubit, GateCNOT, QuantumGate, GateIdentity};

//use std::iter::repeat;
/*
fn linear_toplogy<'a>(double_step: &'a dyn QuantumGate, single_step: &'a dyn QuantumGate, dits: u8, d: u8, identity_step: Option<&'a dyn QuantumGate>, ) {
    let default = GateIdentity::new(d.pow(dits as u32) as usize, d);
    let id = match identity_step {
            Some(v) => v,
            None => &default,
    };
    let v: Vec<&dyn QuantumGate> = vec![(0..dits-1).map(|i| {
        let id_double: Vec<&dyn QuantumGate> = repeat(id).take(dits as usize).collect();
        id_double.insert((i + 1) as usize, double_step);
        let id_single: Vec<&dyn QuantumGate> = repeat(id).take(dits as usize).collect();
        id_single.insert((i + 1) as usize, single_step);
        id_single.insert((i + 1) as usize, single_step);
        let double = GateKronecker::new(id_double);
        let single = GateKronecker::new(id_single);
        let b: &dyn QuantumGate = &GateProduct::new(vec![&double, &single]);
        b
    }).collect()];
    v
}
*/