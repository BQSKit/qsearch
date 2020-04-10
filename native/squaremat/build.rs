use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), std::io::Error> {
    // Homebrew/macOS gcc don't add libgfortran to the rpath,
    // so we manually go prodding around for it here
    if cfg!(target_os = "macos") {
        // First, we get brew prefix, if installed. Fall back to system gcc.
        let (prefix, brew) = match Command::new("brew").arg("--prefix").output() {
            Ok(o) => {
                let s = String::from_utf8(o.stdout).expect("Bad brew --prefix output");
                (s, true)
            }
            Err(_) => (String::from("/usr/local/lib/gcc/"), false),
        };
        let mut gcc_dir = PathBuf::new();
        gcc_dir.push(prefix);
        if brew {
            gcc_dir = fs::read_dir(gcc_dir.join("Cellar/gcc"))?
                .map(|res| res.map(|e| e.path()))
                .filter_map(Result::ok)
                .last()
                .expect("No gcc installed?");
            gcc_dir = gcc_dir.join("lib/gcc");
        }
        if gcc_dir.is_dir() {
            let version_dir = fs::read_dir(gcc_dir)?
                .map(|res| res.map(|e| e.path()))
                .filter_map(Result::ok)
                .last().expect("No directories in prefix?");
            println!("cargo:rustc-link-search={:?}", version_dir);
        }
    }
    Ok(())
}
