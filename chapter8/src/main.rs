mod dct;
mod flood_fill;


fn main() {
    // 1. dct
    let _ = dct::run();

    // 2. flood fill
    let _ = flood_fill::run();
}
