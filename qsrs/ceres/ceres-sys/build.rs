use cmake::Config;
use std::env;

fn main() {
    if env::var("CARGO_FEATURE_STATIC").is_ok() {
        if cfg!(target_os = "windows") {
            #[cfg(target_os = "windows")]
            let ceres = vcpkg::find_package("ceres").unwrap();
            #[cfg(target_os = "windows")]
            let glog = vcpkg::find_package("glog").unwrap();
            #[cfg(target_os = "windows")]
            let gflags = vcpkg::find_package("gflags").unwrap();
            #[cfg(target_os = "windows")]
            let mut conf = cpp_build::Config::new();
            #[cfg(target_os = "windows")]
            for include in ceres
                .include_paths
                .iter()
                .chain(glog.include_paths.iter())
                .chain(gflags.include_paths.iter())
            {
                conf.include(include.to_str().unwrap());
            }
            #[cfg(target_os = "windows")]
            conf.build("src/solve_silent.rs");
            #[cfg(target_os = "windows")]
            println!("cargo:rustc-link-lib=shlwapi")
        } else {
            let ceres = Config::new("ceres-solver")
                .define("EXPORT_BUILD_DIR", "ON")
                .define("CXX11", "ON")
                .define("CXX11_THREADS", "ON")
                .define("BUILD_TESTING", "OFF")
                .define("BUILD_BENCHMARKS", "OFF")
                .define("MINIGLOG", "ON")
                .define("LAPACK", "OFF")
                .define("CUSTOM_BLAS", "OFF")
                .define("SCHUR_SPECIALIZATIONS", "OFF")
                .define("BUILD_EXAMPLES", "OFF")
                .define("LIB_SUFFIX", "")
                .define("SUITESPARSE", "OFF")
                .define("CXSPARSE", "OFF")
                .build();
            println!("cargo:rustc-link-search=native={}/lib", ceres.display());
            println!("cargo:rustc-link-lib=static=ceres");
            cpp_build::Config::new()
                .include(format!("{}/include", ceres.display()))
                .include(format!(
                    "{}/include/ceres/internal/miniglog",
                    ceres.display()
                ))
                .include("/usr/include/eigen3")
                .include("/usr/local/include/eigen3")
                .include("/usr/local/include/eigen")
                .build("src/solve_silent.rs");
        }
    } else {
        println!("cargo:rustc-link-lib=ceres");
        cpp_build::Config::new()
            .include("/usr/include/eigen3")
            .include("/usr/local/include/eigen3")
            .include("/usr/local/include/eigen")
            .build("src/solve_silent.rs");
    }
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if target.contains("win") {
    } else {
        unimplemented!()
    }
}
