mod dct;
mod flood_fill;
mod grab_cut;
mod inpaint;

fn main() {
    // 1. dct
    let _ = dct::run();

    // 2. flood fill
    // let _ = flood_fill::run();

    // 3. grab cut
    // let _ = grab_cut::run();

    // 4. inpaint
    let _ = inpaint::run();
}
