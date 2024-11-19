use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Scalar, Point, Point2f, Size, Vector, Vec3b, TermCriteria},
    imgcodecs,
    imgproc,
    tracking,
    highgui,
    videoio,
    video,
    calib3d,
    features2d,
};

const BASE_PATH: &str = "../data/chapter11/";

pub(crate) fn run() -> Result<()> {
    // 打开视频文件
    let mut cap = videoio::VideoCapture::from_file(&format!("{}vtest.avi", BASE_PATH), videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("请确认输入的视频文件名是否正确");
    }

    // 是否已经计算目标区域直方图标志，0表示没有计算，1表示已经计算
    let mut track_object = 0;

    // 计算直方图和反向直方图相关参数
    let h_size = 16;
    let h_ranges = [0f32, 180f32];
    let ph_ranges = &h_ranges;

    // 初始化各种Mat对象
    let mut frame = Mat::default();
    let mut hsv = Mat::default();
    let mut hue = Mat::default();
    let mut hist = Mat::default();
    let mut hist_img = Mat::zeros(200, 320, opencv::core::CV_8UC3)?.to_mat()?;
    let mut backproj = Mat::default();

    // 读取第一帧并选择跟踪区域
    cap.read(&mut frame)?;
    let mut selection = highgui::select_roi("选择目标跟踪区域", &frame, true, false, true)?;

    loop {
        // 判断是否读取了全部图像
        if !cap.read(&mut frame)? {
            break;
        }

        // 将图像转化成HSV颜色空间
        imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

        // 定义计算直方图和反向直方图相关数据和图像
        let ch = [0, 0];
        hue = Mat::zeros(hsv.rows(), hsv.cols(), hsv.depth())?.to_mat()?;
        opencv::core::mix_channels(
            &Vector::<Mat>::from_elem(hsv.clone(), 1),
            &mut hue,
            &ch,
        )?;

        // 是否已经完成跟踪目标直方图的计算
        if track_object <= 0 {
            // 目标区域的HSV颜色空间
            let roi = Mat::roi(&hue, selection)?.try_clone()?;

            // 计算直方图和直方图归一化
            imgproc::calc_hist(
                &Vector::<Mat>::from_elem(roi, 1),
                &Vector::from_slice(&[0]),
                &Mat::default(),
                &mut hist,
                &Vector::from_slice(&[h_size]),
                &Vector::from_slice(&h_ranges),
                false,
            )?;
            opencv::core::normalize(&hist.clone(), &mut hist, 0.0, 255.0,
                                    opencv::core::NORM_MINMAX, -1, &opencv::core::no_array())?;

            // 将标志设置为1，不再计算目标区域的直方图
            track_object = 1;

            // 重置直方图显示图像
            hist_img = Mat::zeros(200, 320, opencv::core::CV_8UC3)?.to_mat()?;

            // 显示目标区域的直方图
            let bin_w = hist_img.cols() / h_size;
            let mut buf = Mat::zeros(1, h_size, opencv::core::CV_8UC3)?.to_mat()?;

            // 创建颜色条
            for i in 0..h_size {
                let color = Vec3b::from_array([
                    (i as f32 * 180.0 / h_size as f32) as u8,
                    255 as u8,
                    255 as u8,
                ]);
                *buf.at_mut::<Vec3b>(i)? = color;
            }

            imgproc::cvt_color(&buf.clone(), &mut buf, imgproc::COLOR_HSV2BGR, 0)?;

            // 绘制直方图
            for i in 0..h_size {
                let rows = hist_img.rows();
                let val = (hist.at::<f32>(i)? * rows as f32 / 255.0) as i32;

                imgproc::rectangle_points(
                    &mut hist_img,
                    Point::new(i * bin_w, rows),
                    Point::new((i + 1) * bin_w, rows - val),
                    Scalar::new(
                        buf.at::<Vec3b>(i)?[0] as f64,
                        buf.at::<Vec3b>(i)?[1] as f64,
                        buf.at::<Vec3b>(i)?[2] as f64,
                        0.0,
                    ),
                    -1,
                    8,
                    0,
                )?;
            }
        }

        // 计算目标区域的反向直方图
        imgproc::calc_back_project(
            &Vector::<Mat>::from_elem(hue, 1),
            &Vector::from_slice(&[0]),
            &hist,
            &mut backproj,
            &Vector::from_slice(&h_ranges),
            1.0,
        )?;

        // 均值迁移法跟踪目标
        video::mean_shift(
            &backproj,
            &mut selection,
            TermCriteria::new(
                opencv::core::TermCriteria_EPS + opencv::core::TermCriteria_COUNT,
                10,
                1.0,
            )?,
        )?;

        // 在图像中绘制寻找到的跟踪窗口
        imgproc::rectangle(
            &mut frame,
            selection,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            3,
            imgproc::LINE_AA,
            0,
        )?;

        // 显示结果
        highgui::imshow("MeanShift Demo", &frame)?;  // 显示跟踪结果
        highgui::imshow("Histogram", &hist_img)?;    // 显示目标区域直方图

        // 按ESC键退出程序
        let key = highgui::wait_key(50)?;
        if key == 27 {
            break;
        }
    }

    Ok(())
}
