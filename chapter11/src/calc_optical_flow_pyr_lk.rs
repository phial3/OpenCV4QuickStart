use anyhow::{Result, Error, Context};
use opencv::{
    prelude::*,
    core::{Mat, Scalar, Point, Point2f, Size, Vector, TermCriteria},
    imgcodecs,
    imgproc,
    highgui,
    videoio,
    video,
    calib3d,
    features2d,
};
use rand::Rng;

const BASE_PATH: &str = "../data/chapter11/";

pub(crate) fn run() -> Result<()> {
    let mut color_lut = ColorLUT::new();

    // 打开视频文件
    let mut capture = videoio::VideoCapture::from_file(&format!("{}mulballs.mp4", BASE_PATH), videoio::CAP_ANY, )?;

    let mut prev_frame = Mat::default();
    let mut prev_img = Mat::default();

    if !capture.read(&mut prev_frame)? {
        panic!("请确认输入视频文件是否正确");
    }

    imgproc::cvt_color(&prev_frame, &mut prev_img, imgproc::COLOR_BGR2GRAY, 0)?;

    // 角点检测相关参数设置
    let mut points = Vector::<Point2f>::new();
    let quality_level = 0.01;
    let min_distance = 10.0;
    let block_size = 3;
    let use_harris_detector = false;
    let k = 0.04;
    let corners = 5000;

    // 角点检测
    imgproc::good_features_to_track(
        &prev_img,
        &mut points,
        corners,
        quality_level,
        min_distance,
        &Mat::default(),
        block_size,
        use_harris_detector,
        k,
    )?;

    // 稀疏光流检测相关参数设置
    let mut prev_pts = points.clone();
    let mut next_pts = Vector::<Point2f>::new();
    let mut status = Vector::<u8>::new();
    let mut err = Vector::<f32>::new();

    let criteria = TermCriteria::new(
        opencv::core::TermCriteria_COUNT + opencv::core::TermCriteria_EPS,
        30,
        0.01,
    )?;

    // 初始状态的角点
    let mut init_points = points.clone();

    loop {
        let mut next_frame = Mat::default();
        let mut next_img = Mat::default();

        if !capture.read(&mut next_frame)? {
            break;
        }

        highgui::imshow("nextframe", &next_frame)?;

        // 光流跟踪
        imgproc::cvt_color(&next_frame, &mut next_img, imgproc::COLOR_BGR2GRAY, 0)?;

        video::calc_optical_flow_pyr_lk(
            &prev_img,
            &next_img,
            &prev_pts,
            &mut next_pts,
            &mut status,
            &mut err,
            Size::new(31, 31),
            3,
            criteria,
            0,
            0.0,
        )?;

        // 判断角点是否移动，如果不移动就删除
        let mut valid_points = Vector::<Point2f>::new();
        let mut valid_init_points = Vector::<Point2f>::new();
        let mut valid_next_points = Vector::<Point2f>::new();

        for i in 0..next_pts.len() {
            let dist = (prev_pts.get(i)?.x - next_pts.get(i)?.x).abs() +
                (prev_pts.get(i)?.y - next_pts.get(i)?.y).abs();

            if status.get(i)? != 0 && dist > 2.0 {
                valid_points.push(prev_pts.get(i)?);
                valid_init_points.push(init_points.get(i)?);
                valid_next_points.push(next_pts.get(i)?);

                imgproc::circle(
                    &mut next_frame,
                    Point::new(next_pts.get(i)?.x as i32, next_pts.get(i)?.y as i32),
                    3,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    -1,
                    8,
                    0,
                )?;
            }
        }

        prev_pts = valid_points;
        init_points = valid_init_points;
        next_pts = valid_next_points;

        // 绘制跟踪轨迹
        draw_lines(&mut next_frame, &init_points, &next_pts, &mut color_lut)?;
        highgui::imshow("result", &next_frame)?;

        let key = highgui::wait_key(50)?;
        if key == 27 {
            break;
        }

        // 更新角点坐标和前一帧图像
        std::mem::swap(&mut next_pts, &mut prev_pts);
        next_img.copy_to(&mut prev_img)?;

        // 如果角点数目少于30，就重新检测角点
        if init_points.len() < 30 {
            imgproc::good_features_to_track(
                &prev_img,
                &mut points,
                corners,
                quality_level,
                min_distance,
                &Mat::default(),
                block_size,
                use_harris_detector,
                k,
            )?;

            for point in points.iter() {
                init_points.push(point);
                prev_pts.push(point);
            }

            println!("total feature points : {}", prev_pts.len());
        }
    }

    Ok(())
}

// 全局颜色查找表
struct ColorLUT {
    colors: Vec<Scalar>,
}

impl ColorLUT {
    fn new() -> Self {
        ColorLUT { colors: Vec::new() }
    }

    fn ensure_size(&mut self, size: usize) {
        let mut rng = rand::thread_rng();
        while self.colors.len() < size {
            self.colors.push(Scalar::new(
                rng.gen_range(0.0..255.0),
                rng.gen_range(0.0..255.0),
                rng.gen_range(0.0..255.0),
                0.0,
            ));
        }
    }
}

fn draw_lines(
    image: &mut Mat,
    pt1: &Vector<Point2f>,
    pt2: &Vector<Point2f>,
    color_lut: &mut ColorLUT,
) -> opencv::Result<()> {
    color_lut.ensure_size(pt1.len());

    for i in 0..pt1.len() {
        imgproc::line(
            image,
            Point::new(pt1.get(i)?.x as i32, pt1.get(i)?.y as i32),
            Point::new(pt2.get(i)?.x as i32, pt2.get(i)?.y as i32),
            color_lut.colors[i],
            2,
            8,
            0,
        )?;
    }
    Ok(())
}
