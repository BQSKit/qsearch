use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), std::io::Error> {
    // Homebrew/macOS gcc don't add libgfortran to the rpath,
    // so we manually go prodding around for it here
    if cfg!(target_os = "macos") {
        // First, we get brew prefix, if installed. Fall back to system gcc.
        let prefix = String::from_utf8(
            Command::new("brew")
                .arg("--prefix")
                .output()
                .expect("Failed to run brew --prefix")
                .stdout,
        )
        .unwrap()
        .replace("\n", "");
        let mut gcc_dir = PathBuf::new();
        gcc_dir.push(prefix.clone());

        gcc_dir = fs::read_dir(gcc_dir.join("Cellar/gcc"))?
            .map(|res| res.map(|e| e.path()))
            .filter_map(Result::ok)
            .last()
            .expect("No gcc installed?");
        gcc_dir = gcc_dir.join("lib/gcc");
        let version_dir = fs::read_dir(gcc_dir)?
            .map(|res| res.map(|e| e.path()))
            .filter_map(Result::ok)
            .last()
            .expect("No directories in prefix?");
        let mut openblas_dir = PathBuf::new();
        openblas_dir.push(prefix);
        openblas_dir.push("opt/openblas/lib");
        println!("cargo:rustc-link-search={}", version_dir.to_str().unwrap());
        println!("cargo:rustc-link-search={}", openblas_dir.to_str().unwrap());
    }
    Ok(())
}
