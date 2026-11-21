use serde_json::{json, Value};
fn and(a: i32, b: i32, z: i32) -> i32 {
    a * b - 2 * a * z - 2 * b * z - a - b + 2 * z
}
fn xor(a: i32, b: i32, z: i32, w: i32) -> i32 {
    a * b - a * z - 2 * a * w - b * z - 2 * b * w + 2 * z * w - a - b + z + 2 * w
}
fn rows(n: usize) -> impl Iterator<Item = Vec<i32>> {
    (0..1 << n).map(move |i| (0..n).rev().map(|j| (i >> j) & 1).collect())
}
pub fn enumerate() -> (Vec<Value>, Vec<Value>) {
    let gates = [
        ("AND", 3),
        ("OR", 3),
        ("NOT", 2),
        ("COPY", 2),
        ("SET0", 1),
        ("SET1", 1),
        ("XOR", 3),
    ]
    .into_iter()
    .map(|(name, n)| {
        let rows:Vec<_>=rows(n).map(|bits|{
            let s:Vec<_>=bits.iter().map(|x|2*x-1).collect();
            let a=s[0]; let z=s[n-1]; let b=s[n/2];
            let (energy,allowed)=match name {
                "AND"=>(and(a,b,z),bits[2]==(bits[0]&bits[1])),
                "OR"=>(and(-a,-b,-z),bits[2]==(bits[0]|bits[1])),
                "NOT"=>(a*z,a!=z),
                "COPY"=>(-a*z,a==z),
                "SET0"=>(z,z==-1),
                "SET1"=>(-z,z==1),
                "XOR"=>(xor(a,b,z,-1).min(xor(a,b,z,1)),bits[2]==(bits[0]^bits[1])),
                _=>unreachable!(),
            };
            let helper_energies=(name=="XOR").then(||[xor(a,b,z,-1),xor(a,b,z,1)]);
            json!({"bits":bits,"energy":energy,"allowed":allowed,"helper_energies":helper_energies})
        }).collect();
        json!({"gate":name,"rows":rows})
    })
    .collect();
    let circuit = rows(4)
        .map(|bits| {
            let s: Vec<_> = bits.iter().map(|x| 2 * x - 1).collect();
            let terms = [and(s[0], s[1], s[2]), s[2] * s[3], -s[3]];
            json!({"bits":bits,"gate_energies":terms,"energy":terms.iter().sum::<i32>(),
              "allowed":bits[2]==(bits[0]&bits[1])&&bits[3]==1-bits[2]&&bits[3]==1})
        })
        .collect();
    (gates, circuit)
}

#[test]
fn truth_table_floors() {
    let (g, c) = enumerate();
    for (gate, floor) in g.iter().zip([-3, -3, -1, -1, -1, -1, -4]) {
        for row in gate["rows"].as_array().unwrap() {
            let e = row["energy"].as_i64().unwrap();
            if row["allowed"] == true {
                assert_eq!(e, floor);
            } else {
                assert!(e >= floor + 2);
            }
        }
    }
    assert_eq!(c.iter().filter(|r| r["energy"] == -5).count(), 3);
    assert!(c
        .iter()
        .all(|r| (r["energy"] == -5) == (r["allowed"] == true)));
}
