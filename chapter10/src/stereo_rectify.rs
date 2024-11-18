use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use opencv::{
    calib3d,
    core::{Mat, Point, Point2f, Point3f, Scalar, Size, Vector, TermCriteria},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 读取所有图像
    let mut imgLs = Vector::<Mat>::new();
    let mut imgRs = Vector::<Mat>::new();

    let file_l = File::open(BASE_PATH.to_string() + "steroCalibDataL.txt")?;
    let file_r = File::open(BASE_PATH.to_string() + "steroCalibDataR.txt")?;

    let mut finL = BufReader::new(file_l);
    let mut finR = BufReader::new(file_r);

    let mut imgLName = String::new();
    let mut imgRName = String::new();

    while finL.read_line(&mut imgLName)? > 0 && finR.read_line(&mut imgRName)? > 0 {
        imgLName = imgLName.trim().to_string();
        imgRName = imgRName.trim().to_string();

        println!("正在读取图像：{} {}", imgLName, imgRName);
        let imgL = imgcodecs::imread(&(BASE_PATH.to_string() + &imgLName), imgcodecs::IMREAD_COLOR)?;
        let imgR = imgcodecs::imread(&(BASE_PATH.to_string() + &imgRName), imgcodecs::IMREAD_COLOR)?;

        imgLs.push(imgL);
        imgRs.push(imgR);
    }

    let board_size = Size::new(9, 6); // 方格标定板内角点数目（行，列）
    let imgLsPoints = get_imgs_points(imgLs.clone(), board_size)?;
    let imgRsPoints = get_imgs_points(imgRs.clone(), board_size)?;

    // 生成棋盘格每个内角点的空间三维坐标
    let square_size = Size::new(10, 10); // 棋盘格每个方格的真实尺寸
    let mut object_points = Vector::<Vector<Point3f>>::new();

    for _ in 0..imgLsPoints.len() {
        let mut temp_point_set = Vector::<Point3f>::new();
        for j in 0..board_size.height {
            for k in 0..board_size.width {
                let real_point = Point3f::new(
                    (j * square_size.width) as f32,
                    (k * square_size.height) as f32,
                    0.0, // 假设标定板为世界坐标系的z=0平面
                );
                temp_point_set.push(real_point);
            }
        }
        object_points.push(temp_point_set);
    }

    // 图像尺寸
    let i0 = imgLs.get(0)?;
    let image_size = Size::new(i0.cols(), i0.rows());

    // 校准左相机和右相机
    let mut matrix1 = Mat::default();
    let mut dist1 = Mat::default();
    let mut matrix2 = Mat::default();
    let mut dist2 = Mat::default();
    let mut rvecs = Mat::default();
    let mut tvecs = Mat::default();

    calib3d::calibrate_camera(&object_points, &imgLsPoints, image_size, &mut matrix1, &mut dist1, &mut rvecs, &mut tvecs, 0, TermCriteria::default().unwrap())?;
    calib3d::calibrate_camera(&object_points, &imgRsPoints, image_size, &mut matrix2, &mut dist2, &mut rvecs, &mut tvecs, 0, TermCriteria::default().unwrap())?;

    // 进行立体标定
    let mut R = Mat::default();
    let mut T = Mat::default();
    let mut E = Mat::default();
    let mut F = Mat::default();

    calib3d::stereo_calibrate(
        &object_points,
        &imgLsPoints,
        &imgRsPoints,
        &mut matrix1,
        &mut dist1,
        &mut matrix2,
        &mut dist2,
        image_size,
        &mut R,
        &mut T,
        &mut E,
        &mut F,
        calib3d::CALIB_USE_INTRINSIC_GUESS,
        TermCriteria::default()?,
    )?;

    // 计算校正变换矩阵
    let mut R1 = Mat::default();
    let mut R2 = Mat::default();
    let mut P1 = Mat::default();
    let mut P2 = Mat::default();
    let mut Q = Mat::default();

    calib3d::stereo_rectify_def(
        &matrix1,
        &dist1,
        &matrix2,
        &dist2,
        image_size,
        &R,
        &T,
        &mut R1,
        &mut R2,
        &mut P1,
        &mut P2,
        &mut Q,
    )?;

    // 计算校正映射矩阵
    let mut map11 = Mat::default();
    let mut map12 = Mat::default();
    let mut map21 = Mat::default();
    let mut map22 = Mat::default();

    calib3d::init_undistort_rectify_map(
        &matrix1,
        &dist1,
        &R1,
        &P1,
        image_size,
        opencv::core::CV_16SC2,
        &mut map11,
        &mut map12,
    )?;
    calib3d::init_undistort_rectify_map(
        &matrix2,
        &dist2,
        &R2,
        &P2,
        image_size,
        opencv::core::CV_16SC2,
        &mut map21,
        &mut map22,
    )?;

    for i in 0..imgLs.len() {
        // 进行校正映射
        let mut img1r = Mat::default();
        let mut img2r = Mat::default();

        imgproc::remap(&imgLs.get(i)?, &mut img1r, &map11, &map12, imgproc::INTER_LINEAR,opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;
        imgproc::remap(&imgRs.get(i)?, &mut img2r, &map21, &map22, imgproc::INTER_LINEAR,opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;

        // 拼接图像
        let mut arr = Vector::<Mat>::new();
        arr.push(img1r);
        arr.push(img2r);
        let mut result = Mat::default();
        opencv::core::hconcat(&arr, &mut result)?;

        let cols = result.cols();
        // 绘制直线，用于比较同一个内角点y轴是否一致
        let point = Point::new(-1, imgLsPoints.get(i)?.get(0)?.y as i32);
        imgproc::line(&mut result, point, Point::new(cols, point.y), Scalar::new(0.0, 0.0, 255.0, 0.0), 2, 8, 0)?;

        highgui::imshow("校正后结果", &result)?;
        highgui::wait_key(0)?;
    }

    Ok(())
}

fn get_imgs_points(imgs: Vector<Mat>, board_size: Size) -> opencv::Result<Vector<Vector<Point2f>>> {
    let mut points= Vector::<Vector<Point2f>>::new();
    for img in imgs {
        let mut gray1 = Mat::default();
        imgproc::cvt_color(&img, &mut gray1, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut img1_points= Vector::<Point2f>::new();
        calib3d::find_chessboard_corners(&gray1, board_size, &mut img1_points, calib3d::CALIB_CB_ADAPTIVE_THRESH)?;
        calib3d::find4_quad_corner_subpix(&gray1, &mut img1_points, Size::new(5, 5))?;

        points.push(img1_points);
    }
    Ok(points)
}

