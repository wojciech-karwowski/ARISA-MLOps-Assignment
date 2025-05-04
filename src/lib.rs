use pyo3::prelude::*;

#[pyfunction]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

#[pymodule]
fn arisa_dsml(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greet, m)?)?;
    Ok(())
}