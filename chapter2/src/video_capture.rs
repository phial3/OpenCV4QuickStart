use anyhow::Result;
use opencv::{prelude::*, core::Vector, imgproc, features2d, videoio, highgui};

pub(crate) fn run() -> Result<()> {
    let window = "video capture";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // 0 is the default camera
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&capture)? {
        panic!("Unable to open default camera!");
    }

    let mut orb = features2d::ORB::create_def()?;

    loop {
        let mut frame = Mat::default();
        capture.read(&mut frame)?;

        // 显示原始图像
        // if frame.size()?.width > 0 {
        //     highgui::imshow(window, &frame)?;
        // }

        if frame.cols() > 0 {
            // 将彩色图像 frame 转换为灰度图 gray
            let mut gray = Mat::default();
            imgproc::cvt_color_def(&frame, &mut gray, imgproc::COLOR_BGR2GRAY)?;

            // 使用 ORB（Oriented FAST and Rotated BRIEF）算法在灰度图像 gray 中检测特征点（关键点）。这些关键点存储在 kps 向量中
            let mut kps = Vector::new();
            orb.detect_def(&gray, &mut kps)?;

            // 将检测到的关键点 kps 绘制在 gray 图像上，并将结果存储在 display 图像中。
            let mut display = Mat::default();
            features2d::draw_keypoints_def(&gray, &kps, &mut display)?;

            // 将包含绘制关键点的图像 display 显示在窗口中
            highgui::imshow(window, &display)?;
        }

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }
    Ok(())
}