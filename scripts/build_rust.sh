cd native &&  RUSTFLAGS="-C target-cpu=native" maturin build --release -i python3 --manylinux 2010-unchecked && pip install target/wheels/scrs-*.whl -U && cd ..
