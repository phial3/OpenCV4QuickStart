mod dct;
mod dft;
mod flood_fill;
mod grab_cut;
mod inpaint;
mod integral;
mod mul_spectrums;
mod pyr_mean_shift_filtering;
mod watershed;
mod watershed_self;


fn main() {
    // 1. dct
    // let _ = dct::run();

    // 2. flood fill
    // let _ = flood_fill::run();

    // 3. grab cut
    // let _ = grab_cut::run();

    // 4. inpaint
    // let _ = inpaint::run();

    // 5. integral image
    // let _ = integral::run();

    // 6. multiply spectrums
    // let _ = mul_spectrums::run();

    // 7. pyr mean shift filtering
    // let _ = pyr_mean_shift_filtering::run();

    // 8. watershed
    // let _ = watershed::run();

    // 9. watershed self
    // let _ = watershed_self::run();

    // 10. dft
    let _ = dft::run();
}
