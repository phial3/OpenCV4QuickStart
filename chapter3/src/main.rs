extern crate core;

mod create_track_bar;
mod cv_color;
mod deep_shadow_copy;
mod image_concat;
mod image_flip;
mod image_lut;
mod image_resize;
mod logic_operation;
mod max_and_min;
mod mean_and_mean_std_dev;
mod mouse_event;
mod plot;
mod pyramid;
mod split_and_merge;
mod threshold;
mod warp_polar;
mod warp_affine;
mod warp_perspective;
mod find_min_and_max;

fn main() {
    // 1. image concat
    // let _ = image_concat::run();

    // 2. create track bar
    // let _ = create_track_bar::run();

    // 3. cv color
    // let _ = cv_color::run();

    // 4. deep shadow copy
    // let _ = deep_shadow_copy::run();

    // 5. image flip
    // let _ = image_flip::run();

    // 6. image lut
    // let _ = image_lut::run();

    // 7. logic operation
    // let _ = logic_operation::run();

    // 8. max and min
    // let _ = max_and_min::run();

    // 9. mean and mear std dev
    // let _ = mean_and_mean_std_dev::run();

    // 10. mouse event
    // let _ = mouse_event::run();

    // 11. plot
    // let _ = plot::run();

    // 12. pyramid
    // let _ = pyramid::run();

    // 13. image resize
    // let _ = image_resize::run();

    // 14. split and merge
    // let _ = split_and_merge::run();

    // 15. threshold
    // let _ = threshold::run();

    // 16. wrap affine
    // let _ = warp_affine::run();

    // 17. warp perspective
    // let _ = warp_perspective::run();

    // 18. find min and max
    // let _ = find_min_and_max::run();

    // 19. warp polar
    let _ = warp_polar::run();
}
