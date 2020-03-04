cd native &&  RUSTFLAGS="-C target-cpu=native" maturin build --release -i python --manylinux 2010-unchecked && pip install target/wheels/scrs-*.whl -U && cd ..
